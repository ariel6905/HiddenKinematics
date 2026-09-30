"""Geometry helpers: part bounding boxes at a joint position, IK and quaternion utilities (xyzw)."""
import torch
from isaacgym.torch_utils import quat_conjugate, quat_mul, quat_rotate
from pytorch3d.transforms import axis_angle_to_matrix

from .misc import IK_DAMPING


def _bbox_at_joint(bbox, axis_xyz, axis_dir, qpos, is_door):
    """Move rest-pose boxes to joint position ``qpos`` (rotate for doors, translate for drawers)."""
    rot = axis_angle_to_matrix((qpos * axis_dir.T).T)
    canon = (bbox - axis_xyz.reshape(-1, 1, 3)).reshape(-1, 8, 3, 1)
    door = torch.matmul(rot.reshape(-1, 1, 3, 3), canon).reshape(-1, 8, 3) + axis_xyz.reshape(-1, 1, 3)
    drawer = (bbox - axis_xyz.reshape(-1, 1, 3)) + qpos.reshape(-1, 1, 1) * axis_dir.reshape(-1, 1, 3) \
        + axis_xyz.reshape(-1, 1, 3)
    return torch.where(is_door.reshape(-1, 1, 1), door, drawer)


def _object_to_sim(task, bbox):
    """Transform boxes [E, 8, 3] from the object frame to the simulation frame."""
    return torch.matmul(task.object_init_pose_r_matrix_tensor, bbox.reshape(-1, 8, 3, 1)).reshape(-1, 8, 3) \
        + task.object_init_pose_p_tensor


def part_and_handle_bbox(task, qpos):
    """Part and handle boxes [E, 8, 3] in the simulation frame at joint position ``qpos`` [E]."""
    return (_object_to_sim(task, _bbox_at_joint(task.part_bbox_tensor_init, task.part_axis_xyz_tensor_init,
                                                task.part_axis_dir_tensor_init, qpos, task.is_door)),
            _object_to_sim(task, _bbox_at_joint(task.handle_bbox_tensor_init, task.part_axis_xyz_tensor_init,
                                                task.part_axis_dir_tensor_init, qpos, task.is_door)))


def quat_axis(q, axis=0):
    """Rotate the ``axis``-th basis vector by quaternion ``q`` (xyzw)."""
    basis = torch.zeros(q.shape[0], 3, device=q.device)
    basis[:, axis] = 1
    return quat_rotate(q, basis)


def orientation_error(desired, current):
    """Vector part of ``desired * conj(current)`` (xyzw), sign-corrected to the shorter rotation."""
    q_r = quat_mul(desired, quat_conjugate(current))
    return q_r[:, 0:3] * torch.sign(q_r[:, 3]).unsqueeze(-1)


def control_ik(j_eef, device, dpose, num_envs):
    """Damped least-squares IK step: joint deltas [E, dofs] for the pose error ``dpose`` [E, 6, 1]."""
    j_eef_T = torch.transpose(j_eef, 1, 2)
    lmbda = torch.eye(6, device=device) * (IK_DAMPING ** 2)
    return (j_eef_T @ torch.inverse(j_eef @ j_eef_T + lmbda) @ dpose).view(num_envs, -1)


def relative_pose(src, dst):
    """Subtract the position of ``src`` from ``dst``; the other channels of ``dst`` are kept."""
    shape = dst.shape
    p = dst.view(-1, shape[-1])[:, :3] - src.view(-1, src.shape[-1])[:, :3]
    return torch.cat((p, dst.view(-1, shape[-1])[:, 3:]), dim=1).view(*shape)
