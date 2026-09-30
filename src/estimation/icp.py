"""Batched rigid registration (ICP with Kabsch/SVD), vectorized over environments."""
import torch


def icp(src, dst, valid, iters=5, trim=0.1, reg=0.0):
    """Trimmed ICP registering ``src`` [E, Ns, 3] onto ``dst`` [E, Nd, 3]; returns (R, t)."""
    E, dev = src.shape[0], src.device
    I = torch.eye(3, device=dev).expand(E, 3, 3).contiguous()
    R, t, cur = I.clone(), torch.zeros(E, 3, device=dev), src.clone()
    for _ in range(iters):
        dist, idx = torch.cdist(cur, dst).min(dim=-1)                          # [E, Ns]
        match = torch.gather(dst, 1, idx.unsqueeze(-1).expand(-1, -1, 3))       # [E, Ns, 3]
        thr = torch.quantile(dist, 1.0 - trim, dim=1, keepdim=True)
        w = (dist <= thr).float()
        ws = w.sum(1, keepdim=True).clamp(min=1.0)
        c_src = (w.unsqueeze(-1) * cur).sum(1) / ws
        c_dst = (w.unsqueeze(-1) * match).sum(1) / ws
        H = ((cur - c_src.unsqueeze(1)) * w.unsqueeze(-1)).transpose(1, 2) @ (match - c_dst.unsqueeze(1))
        if reg > 0:
            spread = (((cur - c_src.unsqueeze(1)) ** 2) * w.unsqueeze(-1)).sum((1, 2)) / ws.squeeze(-1)
            H = H + reg * spread[:, None, None] * I
        U, _, Vt = torch.linalg.svd(H)
        sign = torch.sign(torch.linalg.det(Vt.transpose(1, 2) @ U.transpose(1, 2)))
        D = I.clone()
        D[:, 2, 2] = sign
        R_step = Vt.transpose(1, 2) @ D @ U.transpose(1, 2)
        t_step = c_dst - (R_step @ c_src.unsqueeze(-1)).squeeze(-1)
        cur = cur @ R_step.transpose(1, 2) + t_step.unsqueeze(1)
        R = R_step @ R
        t = (R_step @ t.unsqueeze(-1)).squeeze(-1) + t_step
    R = torch.where(valid[:, None, None], R, I)
    t = torch.where(valid[:, None], t, torch.zeros_like(t))
    return torch.nan_to_num(R), torch.nan_to_num(t)


def screw_axis(R, t, min_angle_deg):
    """Decompose rigid motions (R, t) into an axis u and a point c on it; returns (u, c, ok)."""
    theta = torch.acos(((R.diagonal(dim1=-2, dim2=-1).sum(-1) - 1) * 0.5).clamp(-1, 1))
    u = torch.stack([R[:, 2, 1] - R[:, 1, 2],
                     R[:, 0, 2] - R[:, 2, 0],
                     R[:, 1, 0] - R[:, 0, 1]], dim=-1)
    u = u / (u.norm(dim=-1, keepdim=True) + 1e-9)
    I = torch.eye(3, device=R.device).expand_as(R)
    c = (torch.linalg.pinv(I - R) @ t.unsqueeze(-1)).squeeze(-1)
    ok = theta > (min_angle_deg * 3.14159265 / 180.0)
    return torch.nan_to_num(u), torch.nan_to_num(c), ok
