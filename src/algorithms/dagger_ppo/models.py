"""Networks: the vision-based student policy (sparse U-Net encoder + MLP) and the PPO critic."""
import torch
import torch.nn as nn

from ..ppo_utils.backbone import SparseUNetEncoder


class StudentPolicy(nn.Module):
    """Student policy: action = MLP([state, encoder(point cloud with flow)])."""

    def __init__(self, state_dim, action_dim, channels=(16, 64, 112), hidden=(512, 512, 64)):
        super().__init__()
        self.backbone = SparseUNetEncoder(9, channels)
        layers, d = [], state_dim + channels[-1]
        for h in hidden:
            layers += [nn.Linear(d, h), nn.ELU()]
            d = h
        layers += [nn.Linear(d, action_dim)]
        self.head = nn.Sequential(*layers)

    def forward(self, pc, state):
        return self.head(torch.cat([state, self.backbone(pc.contiguous())], dim=-1))


def load_student(bc_ckpt, device, policy_ckpt=None):
    """Build the student from a BC checkpoint, optionally with DAgger+PPO weights; returns (policy, a_mean, a_std)."""
    bc = torch.load(bc_ckpt, map_location="cpu")
    policy = StudentPolicy(int(bc["state_dim"]), int(bc["action_dim"]))
    policy.load_state_dict(bc["policy"])
    if policy_ckpt:
        policy.load_state_dict(torch.load(policy_ckpt, map_location="cpu")["policy"])
    return policy.to(device), bc["a_mean"].float().to(device), bc["a_std"].float().to(device)


class Critic(nn.Module):
    """Value function on the policy's proprioceptive state."""

    def __init__(self, state_dim, hidden=(512, 512, 64)):
        super().__init__()
        dims, layers = [state_dim, *hidden], []
        for a, b in zip(dims[:-1], dims[1:]):
            layers += [nn.Linear(a, b), nn.ELU()]
        self.net = nn.Sequential(*layers, nn.Linear(dims[-1], 1))

    def forward(self, s):
        return self.net(s).squeeze(-1)
