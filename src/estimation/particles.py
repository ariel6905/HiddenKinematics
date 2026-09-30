"""Batched particle set over joint hypotheses (axis direction U, pivot P, type T, weight W)."""
import torch

from .prior import PRISMATIC, REVOLUTE


class ParticleSet:
    """Particles (type, axis, pivot, weight) of E environments x K slots."""

    def __init__(self, num_envs, device, max_k=256):
        self.E, self.K, self.device = num_envs, max_k, device
        self.U = torch.zeros(num_envs, max_k, 3, device=device)
        self.P = torch.zeros(num_envs, max_k, 3, device=device)
        self.T = torch.full((num_envs, max_k), 2, dtype=torch.long, device=device)
        self.W = torch.zeros(num_envs, max_k, device=device)
        self.valid = torch.zeros(num_envs, max_k, dtype=torch.bool, device=device)

    def reset(self, particles):
        """Load ``particles[e] = (U, P, T, W)`` (numpy arrays) for every env ``e``; weights are normalized."""
        self.U.zero_(); self.P.zero_(); self.T.fill_(2); self.W.zero_(); self.valid.zero_()
        for e, (U, P, T, W) in enumerate(particles):
            k = min(len(W), self.K)
            self.U[e, :k] = torch.as_tensor(U[:k], dtype=torch.float32, device=self.device)
            self.P[e, :k] = torch.as_tensor(P[:k], dtype=torch.float32, device=self.device)
            self.T[e, :k] = torch.as_tensor(T[:k], dtype=torch.long, device=self.device)
            w = torch.as_tensor(W[:k], dtype=torch.float32, device=self.device)
            self.W[e, :k] = w / w.sum().clamp(min=1e-12)
            self.valid[e, :k] = True

    def add(self, e, slot, u, p, joint_type, weight):
        """Write one candidate into ``slot`` of env ``e``; returns True if the slot was free."""
        self.U[e, slot] = u
        self.P[e, slot] = p
        self.T[e, slot] = joint_type
        if bool(self.valid[e, slot]):
            return False
        self.W[e, slot] = weight
        self.valid[e, slot] = True
        return True

    def normalize(self, e):
        """Zero the weights of invalid slots of env ``e`` and renormalize."""
        self.W[e] = self.W[e] * self.valid[e].float()
        self.W[e] = self.W[e] / self.W[e].sum().clamp_min(1e-12)

    def free_slots(self, e):
        """Indices of the unused slots of env ``e``."""
        return (~self.valid[e]).nonzero(as_tuple=True)[0]

    def median_weight(self, e, default):
        """Median weight of the valid particles of env ``e`` (``default`` if there are none)."""
        if bool(self.valid[e].any()):
            return self.W[e][self.valid[e]].median()
        return torch.tensor(default, device=self.device)

    def is_prismatic(self):
        """Joint type by belief mass: prismatic iff its particles outweigh the revolute ones."""
        W = self.W.masked_fill(~self.valid, 0.0)
        return (W * (self.T == PRISMATIC).float()).sum(1) > (W * (self.T == REVOLUTE).float()).sum(1)

    def mean_axis_all(self):
        """Weighted mean axis over the valid particles, sign-aligned to the best one."""
        Wk = self.W.unsqueeze(-1)
        m = self.valid.unsqueeze(-1).float()
        ref = self.U[torch.arange(self.E, device=self.device), self.W.argmax(dim=1)]
        sgn = torch.sign((self.U * ref.unsqueeze(1)).sum(dim=-1, keepdim=True) + 1e-9)
        u = (Wk * self.U * sgn * m).sum(dim=1)
        u = u / (u.norm(dim=-1, keepdim=True) + 1e-9)
        return torch.nan_to_num(u)

    def mean_axis_of_type(self, prismatic):
        """Sign-aligned weighted mean axis over the particles of the selected type per env."""
        sel = (torch.where(prismatic.unsqueeze(1), self.T == PRISMATIC, self.T == REVOLUTE) & self.valid).float()
        Ws = self.W * sel
        ref = self.U[torch.arange(self.E, device=self.device), Ws.argmax(dim=1)]
        sgn = torch.sign((self.U * ref.unsqueeze(1)).sum(-1, keepdim=True) + 1e-9)
        u = (Ws.unsqueeze(-1) * self.U * sgn).sum(dim=1)
        return torch.nan_to_num(u / (u.norm(dim=-1, keepdim=True) + 1e-9))

    def best_axis(self, sign_ref):
        """Axis of the single highest-weight particle, sign-aligned to ``sign_ref``."""
        W = self.W.masked_fill(~self.valid, float("-inf"))
        u = self.U[torch.arange(self.E, device=self.device), W.argmax(dim=1)]
        return u * torch.sign((u * sign_ref).sum(-1, keepdim=True) + 1e-9)
