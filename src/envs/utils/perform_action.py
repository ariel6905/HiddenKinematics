"""Map actions (gripper pose and finger targets) to joint targets with damped least-squares IK.

The target quaternion is wxyz (pytorch3d) while IsaacGym uses xyzw; the policies were trained this way.
"""
import pytorch3d.transforms as pt
import torch
from isaacgym import gymtorch

from .misc import CLIP_ACTIONS
from .compute import control_ik, orientation_error


def apply_actions(task, actions):
    """Convert ``actions`` [E, 8] to DOF targets and send them to the simulator."""
    task.pos_act[:] = 0
    task.eff_act[:] = 0
    task.vel_act[:] = 0
    actions = torch.clamp(actions, -CLIP_ACTIONS, CLIP_ACTIONS)
    dof_pos = task.franka_dof_tensor[:, :, 0]
    pos_err = actions[:, :3] - task.hand_rigid_body_tensor[:, :3]
    target_rot = pt.matrix_to_quaternion(pt.axis_angle_to_matrix(actions[:, 3:6]))
    rot_err = orientation_error(target_rot, task.hand_rigid_body_tensor[:, 3:7])
    dpose = torch.cat([pos_err, rot_err], -1).unsqueeze(-1)
    delta = control_ik(task.jacobian_tensor[:, task.hand_rigid_body_index - 1, :, :-2], task.device, dpose, task.num_envs)
    task.pos_act[:, :-3] = dof_pos.squeeze(-1)[:, :-2] + delta
    task.pos_act[:, -3:-1] = actions[:, -2:]

    task.pos_act_all[task.dof_state_mask] = task.pos_act
    task.vel_act_all[task.dof_state_mask] = task.vel_act
    task.eff_act_all[task.dof_state_mask] = task.eff_act
    task.gym.set_dof_position_target_tensor(task.sim, gymtorch.unwrap_tensor(task.pos_act_all))
    task.gym.set_dof_velocity_target_tensor(task.sim, gymtorch.unwrap_tensor(task.vel_act_all))
    task.gym.set_dof_actuation_force_tensor(task.sim, gymtorch.unwrap_tensor(task.eff_act_all))
