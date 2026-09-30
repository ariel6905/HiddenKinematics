"""State-based teacher policies (PPO on the 64-d privileged state, as in PartManip)."""
from collections import OrderedDict

import torch
import torch.nn as nn


def load_state_teacher(ckpt_path, device):
    """Rebuild the deterministic teacher actor; returns (actor, input_dim)."""
    sd = torch.load(ckpt_path, map_location=device)["model_state_dict"]
    idxs = sorted({int(k.split(".")[1]) for k in sd if k.startswith("actor_mlp.") and k.endswith(".weight")})
    mods = []
    for n, i in enumerate(idxs):
        w = sd[f"actor_mlp.{i}.weight"]
        mods.append((str(i), nn.Linear(w.shape[1], w.shape[0])))
        if n < len(idxs) - 1:
            mods.append((str(i + 1), nn.ELU()))
    actor = nn.Sequential(OrderedDict(mods))
    actor.load_state_dict({k[len("actor_mlp."):]: v for k, v in sd.items() if k.startswith("actor_mlp.")})
    return actor.to(device).eval(), int(sd["actor_mlp.0.weight"].shape[1])


class DoorDrawerTeacher:
    """Route every env to the door or the drawer teacher; actions are in the simulation frame."""

    def __init__(self, env, door_ckpt, drawer_ckpt, device):
        self.env = env
        self.door, in_dim = load_state_teacher(door_ckpt, device)
        self.drawer, in_dim2 = load_state_teacher(drawer_ckpt, device)
        assert in_dim == in_dim2 == env.state_dim, (in_dim, in_dim2, env.state_dim)

    @torch.no_grad()
    def __call__(self, obs):
        a_door = self.env.uncanonicalize(self.door(obs.state))
        a_drawer = self.env.uncanonicalize(self.drawer(obs.state))
        return torch.where(self.env.is_door.unsqueeze(-1), a_door, a_drawer)
