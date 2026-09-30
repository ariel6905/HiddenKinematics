"""Observations: proprioceptive and privileged state vectors and the fused camera point cloud."""
from dataclasses import dataclass
from typing import Optional

import torch
from isaacgym.gymtorch import wrap_tensor
from pointnet2_ops import pointnet2_utils
from pytorch3d.transforms import quaternion_apply, quaternion_multiply

from .compute import part_and_handle_bbox, relative_pose


@dataclass
class Observations:
    """Observation batch returned by the environment."""

    state: torch.Tensor                      # Privileged state (state-teacher input), 64-d
    obs: torch.Tensor                        # Observation vector of the visual student, 41-d
    points: Optional[torch.Tensor] = None    # [E, N, 7] point cloud: xyz, rgb, segmentation label


def refresh_tensors(task):
    """Refresh the root, DOF, rigid-body and Jacobian tensors from the simulator."""
    task.gym.refresh_actor_root_state_tensor(task.sim)
    task.gym.refresh_dof_state_tensor(task.sim)
    task.gym.refresh_rigid_body_state_tensor(task.sim)
    task.gym.refresh_jacobian_tensors(task.sim)


def observe(task):
    """Refresh the simulator state and build ``task.obs_buf``."""
    refresh_tensors(task)
    obs = state_observation(task)
    if task.use_pc:
        obs.points = point_cloud(task)
    task.obs_buf = obs
    return obs


def state_observation(task):
    """Build the 64-d privileged state and the 41-d student observation."""
    task.dof_state_tensor_all = wrap_tensor(task.gym.acquire_dof_state_tensor(task.sim))
    task.rigid_body_tensor_all = wrap_tensor(task.gym.acquire_rigid_body_state_tensor(task.sim))
    task.dof_state_tensor_used = task.dof_state_tensor_all[task.dof_state_mask]
    task.rigid_body_tensor_used = task.rigid_body_tensor_all[task.rigid_state_mask]
    task.hand_rigid_body_tensor = task.rigid_body_tensor_used[:, task.hand_rigid_body_index, :]
    task.franka_dof_tensor = task.dof_state_tensor_used[:, :task.franka_num_dofs, :]
    task.cabinet_dof_tensor = task.dof_state_tensor_used[:, task.cabinet_dof_index, :]
    task.cabinet_dof_tensor_spec = task.cabinet_dof_tensor.view(task.cabinet_num, task.env_per_asset, -1)
    part_bbox, handle_bbox = part_and_handle_bbox(task, task.cabinet_dof_tensor[:, 0])

    # Canonical handle frame; the orientation feature mixes xyzw and wxyz (as trained)
    hand_pose = relative_pose(task.franka_root_tensor, task.hand_rigid_body_tensor).view(task.env_num, -1)
    hand_pose[:, :3] += task.franka_root_tensor[:, :3]
    hand_pose[:, :3] -= task.canon_center
    hand_pose[:, :3] = quaternion_apply(task.canon_quaternion_rot, hand_pose[:, :3])
    hand_pose[:, 3:7] = quaternion_multiply(task.canon_quaternion_rot, hand_pose[:, 3:7])
    handle_bbox = quaternion_apply(task.canon_quaternion_rot.view(-1, 1, 4), handle_bbox - task.canon_center.view(-1, 1, 3))
    part_bbox = quaternion_apply(task.canon_quaternion_rot.view(-1, 1, 4), part_bbox - task.canon_center.view(-1, 1, 3))
    root = quaternion_apply(task.canon_quaternion_rot.view(-1, 4), task.franka_root_tensor[:, :3] - task.canon_center)

    bbox_edges = (handle_bbox[:, 0] - handle_bbox[:, 4], handle_bbox[:, 1] - handle_bbox[:, 0],
                  handle_bbox[:, 3] - handle_bbox[:, 0], (handle_bbox[:, 0] + handle_bbox[:, 6]) / 2,
                  part_bbox[:, 0] - part_bbox[:, 4], part_bbox[:, 1] - part_bbox[:, 0],
                  part_bbox[:, 3] - part_bbox[:, 0], (part_bbox[:, 0] + part_bbox[:, 6]) / 2)
    # Rest-pose part center (door: midpoint of box corners 1 and 3, drawer: 2 and 5), simulation frame
    part_center = torch.where(task.is_door.view(-1, 1),
                              (task.init_part_bbox_tensor[:, 1, :] + task.init_part_bbox_tensor[:, 3, :]) / 2,
                              (task.init_part_bbox_tensor[:, 2, :] + task.init_part_bbox_tensor[:, 5, :]) / 2)

    robot_qpose = (2 * (task.franka_dof_tensor[:, :, 0] - task.franka_dof_lower_limits_tensor[:])
                   / (task.franka_dof_upper_limits_tensor[:] - task.franka_dof_lower_limits_tensor[:])) - 1
    robot_qvel = task.franka_dof_tensor[:, :, 1]
    hand_root_pose = torch.cat((root, hand_pose), dim=1)

    state = torch.cat((robot_qpose, robot_qvel, task.cabinet_dof_tensor, hand_root_pose, *bbox_edges), dim=1)
    obs = torch.cat((robot_qpose, robot_qvel, hand_root_pose, part_center), dim=1)
    return Observations(state=state, obs=obs)


# ---------------------------------------------------------------------------------------- Point cloud
def point_cloud(task):
    """Back-project the three depth cameras, crop the workspace and farthest-point sample
    ``num_points`` points per env. Channels: xyz (env frame), rgb in [0, 1], segmentation label."""
    pc_cfg = task.cfg["point_cloud"]
    task.gym.render_all_camera_sensors(task.sim)
    torch.cuda.synchronize()
    task.gym.start_access_image_tensors(task.sim)
    depth = torch.stack([torch.stack(i) for i in task.camera_depth_tensor_list])
    rgb = torch.stack([torch.stack(i) for i in task.camera_rgb_tensor_list])
    seg = torch.stack([torch.stack(i) for i in task.camera_seg_tensor_list])
    view_inv = torch.stack([torch.stack(i) for i in task.camera_view_matrix_inv_list])
    proj = torch.stack([torch.stack(i) for i in task.camera_proj_matrix_list])
    clouds = [_depth_to_points(depth[:, c], rgb[:, c], seg[:, c], view_inv[:, c], proj[:, c], task.camera_u2,
                               task.camera_v2, task.camera_props.width, task.camera_props.height,
                               pc_cfg["max_depth"], task.device, pc_cfg["z_max"], pc_cfg["z_min"])
              for c in range(3)]
    points = torch.cat([c[0] for c in clouds], dim=1)
    points[:, :, :3] -= task.env_origin.view(task.num_envs, 1, 3)
    valid = torch.cat([c[1] for c in clouds], dim=1) * (points[:, :, 0] > pc_cfg["x_min"]) \
        * (points[:, :, 0] < pc_cfg["x_max"]) * (points[:, :, 1] < pc_cfg["y_max"]) * (points[:, :, 1] > pc_cfg["y_min"])

    per_env, start, valid_points = [], 0, points[valid]
    n_pre = pc_cfg["num_presample"]
    for n in valid.sum(1):
        env_points = valid_points[start:start + n]
        start += n
        if env_points.shape[0] == 0:          # Nothing visible (object outside every camera frustum)
            per_env.append(torch.zeros((n_pre, valid_points.shape[-1]), device=task.device))
        else:
            ids = torch.randint(0, env_points.shape[0], (n_pre,), device=task.device, dtype=torch.long)
            per_env.append(env_points[ids])
    batch = torch.stack(per_env)
    idx = pointnet2_utils.furthest_point_sample(batch[:, :, :3].contiguous().cuda(), pc_cfg["num_points"]).long().to(task.device)
    idx = idx.view(*idx.shape, 1).repeat_interleave(batch.shape[-1], dim=2)
    sampled = torch.gather(batch, dim=1, index=idx)
    task.gym.end_access_image_tensors(task.sim)
    return sampled


def _depth_to_points(depth, rgb, seg, view_inv, proj, u, v, width, height, depth_bar, device, z_p_bar, z_n_bar):
    """Back-project one camera per env to [E, H*W, 7] world-frame points and a validity mask."""
    batch = depth.shape[0]
    depth = depth.to(device)
    rgb = rgb.to(device) / 255.0
    seg = seg.to(device)
    fu = 2 / proj[:, 0, 0]
    fv = 2 / proj[:, 1, 1]
    Z = depth
    X = -(u.view(1, u.shape[-2], u.shape[-1]) - width / 2) / width * Z * fu.view(-1, 1, 1)
    Y = (v.view(1, v.shape[-2], v.shape[-1]) - height / 2) / height * Z * fv.view(-1, 1, 1)
    valid_depth = Z.view(batch, -1) > -depth_bar
    pos = torch.cat((X.view(batch, 1, -1), Y.view(batch, 1, -1), Z.view(batch, 1, -1),
                     torch.ones((batch, 1, X.view(batch, 1, -1).shape[-1]), device=device),
                     rgb[..., 0].view(batch, 1, -1), rgb[..., 1].view(batch, 1, -1), rgb[..., 2].view(batch, 1, -1),
                     seg.view(batch, 1, -1)), dim=1).permute(0, 2, 1)
    pos[..., 0:4] = pos[..., 0:4] @ view_inv
    points = pos[..., [0, 1, 2, 4, 5, 6, 7]]
    valid = valid_depth * (pos[..., 2] < z_p_bar) * (pos[..., 2] > z_n_bar)
    return points, valid
