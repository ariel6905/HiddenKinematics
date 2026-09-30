"""Label-free movable-part segmentation: box seed at reset, ICP propagation afterwards."""
import torch

from .icp import icp


def closed_movable_obb(task, device):
    """Oriented boxes [E, 2, 8, 3] of the closed movable part (panel and handle), simulation frame."""
    Rm = task.object_init_pose_r_matrix_tensor.to(device).float()
    p0 = task.object_init_pose_p_tensor.to(device).float().reshape(1, 1, 3)

    def to_world(bb):
        return torch.matmul(Rm, bb.reshape(-1, 8, 3, 1)).reshape(-1, 8, 3) + p0

    return torch.stack([to_world(task.part_bbox_tensor_init.to(device).float()),
                        to_world(task.handle_bbox_tensor_init.to(device).float())], dim=1)


def points_in_obb(pc, obb, margin=0.05):
    """Mask [E, N] of the points ``pc`` [E, N, 3] inside any of the K boxes ``obb`` [E, K, 8, 3]."""
    E, N = pc.shape[:2]
    mask = torch.zeros(E, N, dtype=torch.bool, device=pc.device)
    for k in range(obb.shape[1]):
        b0 = obb[:, k, 0]
        edges = (obb[:, k, 1] - b0, obb[:, k, 3] - b0, obb[:, k, 4] - b0)
        v = pc - b0.unsqueeze(1)
        inside = torch.ones(E, N, dtype=torch.bool, device=pc.device)
        for e in edges:
            s = (v * e.unsqueeze(1)).sum(-1) / (e.pow(2).sum(-1, keepdim=True) + 1e-9)
            inside &= (s >= -margin) & (s <= 1.0 + margin)
        mask |= inside
    return mask


def sample_points(pc, mask, n):
    """Sample ``n`` masked points per env with replacement; returns ``(points [E, n, 3], valid [E])``."""
    E = pc.shape[0]
    out = torch.zeros(E, n, 3, device=pc.device)
    valid = torch.zeros(E, dtype=torch.bool, device=pc.device)
    for e in range(E):
        idx = mask[e].nonzero(as_tuple=True)[0]
        if idx.numel() > 0:
            out[e] = pc[e, idx[torch.randint(idx.numel(), (n,), device=pc.device)]]
            valid[e] = True
    return out, valid


class MovableTracker:
    """Geometric seed at reset and rigid ICP label propagation afterwards."""

    def __init__(self, device, n_mov=512, gate=0.08, tau=0.05, icp_iters=8, obb_margin=0.05):
        self.device = device
        self.n_mov, self.gate, self.tau = n_mov, gate, tau
        self.icp_iters, self.obb_margin = icp_iters, obb_margin
        self._prev_mov = None

    def reset(self, task, pc):
        """Seed the movable set from the closed-part boxes; returns the mask [E, N]."""
        pc = pc.to(self.device).float()
        mask = points_in_obb(pc, closed_movable_obb(task, self.device), self.obb_margin)
        self._prev_mov, _ = sample_points(pc, mask, self.n_mov)
        return mask

    def step(self, pc):
        """Propagate the movable set to ``pc`` with ICP; returns the mask [E, N]."""
        pc = pc.to(self.device).float()
        near_prev = torch.cdist(pc, self._prev_mov).min(-1).values
        target, valid = sample_points(pc, near_prev < self.gate, self.n_mov)
        R, t = icp(self._prev_mov, target, valid, iters=self.icp_iters, trim=0.1)
        moved = self._prev_mov @ R.transpose(1, 2) + t.unsqueeze(1)
        mask = torch.cdist(pc, moved).min(-1).values < self.tau
        self._prev_mov = moved
        return mask
