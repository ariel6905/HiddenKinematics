"""Reward, success and termination for opening doors and drawers."""
import torch

from .compute import part_and_handle_bbox, quat_axis


def compute_reward(task):
    """Set ``task.rew_buf``, ``task.reward_terms`` and the reset and success flags."""
    hand_rot = task.hand_rigid_body_tensor[..., 3:7]
    hand_grip_dir = quat_axis(hand_rot, 2)
    hand_sep_dir = quat_axis(hand_rot, 1)
    hand_down_dir = quat_axis(hand_rot, 0)
    lfinger = task.rigid_body_tensor_used[:, task.hand_lfinger_rigid_body_index][:, 0:3] + hand_grip_dir * 0.1
    rfinger = task.rigid_body_tensor_used[:, task.hand_rfinger_rigid_body_index][:, 0:3] + hand_grip_dir * 0.1
    task.part_bbox_tensor, task.handle_bbox_tensor = part_and_handle_bbox(task, task.cabinet_dof_tensor[:, 0])
    cfg = task.cfg["reward"]

    handle = task.handle_bbox_tensor
    handle_out = handle[:, 0] - handle[:, 4]
    handle_long = handle[:, 1] - handle[:, 0]
    handle_short = handle[:, 3] - handle[:, 0]
    handle_mid = (handle[:, 0] + handle[:, 6]) / 2
    handle_shortest = torch.min(torch.min(torch.norm(handle_out, dim=-1), torch.norm(handle_long, dim=-1)),
                                torch.norm(handle_short, dim=-1))
    handle_out /= torch.norm(handle_out, dim=1, keepdim=True)
    handle_long /= torch.norm(handle_long, dim=1, keepdim=True)
    handle_short /= torch.norm(handle_short, dim=1, keepdim=True)

    # Gripper axes vs. handle axes (gripper axes 1 and 0 may point either way along the handle)
    dot1 = (-hand_grip_dir * handle_out).sum(dim=-1)
    dot2 = torch.max((hand_sep_dir * handle_short).sum(dim=-1), (-hand_sep_dir * handle_short).sum(dim=-1))
    dot3 = torch.max((hand_down_dir * handle_long).sum(dim=-1), (-hand_down_dir * handle_long).sum(dim=-1))
    rot_reward = (torch.sign(dot1) * dot1 ** 2 + 0.5 * (torch.sign(dot2) * dot2 ** 2 + torch.sign(dot3) * dot3 ** 2)) / 2

    finger_mid = (lfinger + rfinger) / 2
    dist_mid = torch.norm(finger_mid - handle_mid, dim=-1)
    reach_reward = -dist_mid
    # Two-stage: the reaching term fades out as the part opens (faster for larger decouple_coef)
    reach_reward *= (dist_mid >= handle_shortest / 2) * torch.clip(
        (task.cabinet_dof_target - task.cabinet_dof_tensor[:, 0] * task.decouple_coef) / task.cabinet_dof_target, min=0)

    hit = (task.cabinet_dof_tensor[:, 0] >= cfg["sparse_threshold"]).float()
    first = hit * (1.0 - task.sparse_paid)
    task.sparse_paid = torch.clamp(task.sparse_paid + hit, max=1.0)
    terms = {"rot": rot_reward * cfg["rot"],
             "reach": reach_reward * cfg["reach"],
             "open": task.cabinet_dof_tensor[:, 0] * cfg["open"],
             "sparse": first * cfg["sparse_bonus"]}
    task.rew_buf = terms["rot"] + terms["reach"] + terms["open"] + terms["sparse"]
    task.reward_terms = {k: v.detach() for k, v in terms.items()}

    # Success: joint within 0.01 of the target (open_proportion) or beyond
    success = ((task.success_dof_states.view(task.cabinet_num, -1) - task.cabinet_dof_tensor_spec[:, :, 0]).view(-1) < 0.01)
    task.reset_buf = task.reset_buf | (task.progress_buf >= task.max_episode_length - 1)
    task.success_buf = task.success_buf | success
