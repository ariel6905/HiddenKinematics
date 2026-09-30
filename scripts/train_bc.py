"""Behavior cloning of the vision-based policy on the teacher demonstrations.

  python scripts/train_bc.py --demo_dirs data/demos/door data/demos/drawer
"""
import argparse
import glob
import os

import numpy as np
import torch
import torch.nn as nn
from torch.utils.data import DataLoader, Dataset

from src.algorithms.dagger_ppo.models import StudentPolicy  # noqa: E402


def load_episodes(demo_dirs, success_rate, seed):
    """Load per-asset .npy demos into episodes ``{point_cloud, state, action}``, filtered by success."""
    kept, failed = [], []
    for d in demo_dirs:
        files = sorted(glob.glob(os.path.join(d, "*.npy")))
        assert files, f"no demonstrations in {d}"
        for f in files:
            demo = np.load(f, allow_pickle=True).item()
            obs = np.asarray(demo["observations"], np.float32)
            act = np.asarray(demo["actions"], np.float32)
            pcs = np.asarray(demo["pcs"], np.float32)
            succ = np.asarray(demo["success"], np.float32)
            for j in range(obs.shape[0]):
                ep = {"point_cloud": pcs[j][..., :9], "state": obs[j], "action": act[j]}         # drop the label
                (kept if succ[j] >= 0.5 else failed).append(ep)
    n_succ = len(kept)
    if success_rate <= 0:                                 # keep all
        n_fail = len(failed)
    else:
        n_fail = min(len(failed), 0 if success_rate >= 1.0 else int(round(n_succ * (1.0 - success_rate) / success_rate)))
    assert n_succ + n_fail > 0, "no demonstrations left after the success filter (use --success_rate 0 to keep all)"
    for i in np.random.RandomState(seed).permutation(len(failed))[:n_fail]:
        kept.append(failed[i])
    print(f"[bc] {len(kept)} episodes ({n_succ} successful + {n_fail} of {len(failed)} failed)", flush=True)
    return kept


class StepDataset(Dataset):
    """One sample per (episode, timestep)."""

    def __init__(self, episodes):
        self.episodes = episodes
        self.index = [(i, t) for i, ep in enumerate(episodes) for t in range(ep["action"].shape[0])]

    def __len__(self):
        return len(self.index)

    def __getitem__(self, k):
        i, t = self.index[k]
        ep = self.episodes[i]
        return (torch.from_numpy(ep["point_cloud"][t]), torch.from_numpy(ep["state"][t]),
                torch.from_numpy(ep["action"][t]))


def main():
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--demo_dirs", nargs="+", default=["data/demos/door", "data/demos/drawer"])
    ap.add_argument("--out_dir", default="runs/bc")
    ap.add_argument("--success_rate", type=float, default=0.8, help="success rate of the kept demos (0: keep all)")
    ap.add_argument("--val_ratio", type=float, default=0.05)
    ap.add_argument("--epochs", type=int, default=200)
    ap.add_argument("--batch_size", type=int, default=32)
    ap.add_argument("--lr", type=float, default=1e-4)
    ap.add_argument("--weight_decay", type=float, default=1e-6)
    ap.add_argument("--save_from", type=int, default=50, help="keep a checkpoint every 10 epochs from this epoch")
    ap.add_argument("--workers", type=int, default=4)
    ap.add_argument("--seed", type=int, default=42)
    args = ap.parse_args()
    torch.manual_seed(args.seed); np.random.seed(args.seed)
    device = "cuda:0"
    os.makedirs(args.out_dir, exist_ok=True)

    episodes = load_episodes(args.demo_dirs, args.success_rate, args.seed)
    actions = np.concatenate([ep["action"] for ep in episodes], 0)
    a_mean = torch.tensor(actions.mean(0), device=device)
    a_std = torch.tensor(actions.std(0) + 1e-6, device=device)
    n_val = min(max(1, round(len(episodes) * args.val_ratio)), len(episodes) - 1)
    val = set(np.random.default_rng(seed=args.seed).choice(len(episodes), size=n_val, replace=False).tolist())
    train = StepDataset([ep for i, ep in enumerate(episodes) if i not in val])
    loader = DataLoader(train, batch_size=args.batch_size, shuffle=True, num_workers=args.workers,
                        drop_last=True, pin_memory=True, persistent_workers=args.workers > 0)
    val_loader = DataLoader(StepDataset([episodes[i] for i in sorted(val)]), batch_size=args.batch_size)
    state_dim, action_dim = episodes[0]["state"].shape[-1], episodes[0]["action"].shape[-1]
    policy = StudentPolicy(state_dim, action_dim).to(device)
    opt = torch.optim.AdamW(policy.parameters(), lr=args.lr, weight_decay=args.weight_decay)
    print(f"[bc] state {state_dim} action {action_dim} samples {len(train)} train / {len(val_loader.dataset)} val", flush=True)

    for epoch in range(args.epochs):
        policy.train()
        losses = []
        for pc, st, ac in loader:
            pred = policy(pc.to(device, non_blocking=True), st.to(device, non_blocking=True))
            loss = nn.functional.mse_loss(pred, (ac.to(device, non_blocking=True) - a_mean) / a_std)
            opt.zero_grad(); loss.backward(); opt.step()
            losses.append(loss.item())
        policy.eval()
        with torch.no_grad():
            val_losses = [nn.functional.mse_loss(policy(pc.to(device), st.to(device)), (ac.to(device) - a_mean) / a_std).item()
                          for pc, st, ac in val_loader]
        print(f"[bc] epoch {epoch:4d} mse {np.mean(losses):.5f} val {np.mean(val_losses):.5f}", flush=True)
        ck = {"policy": policy.state_dict(), "a_mean": a_mean.cpu(), "a_std": a_std.cpu(), "epoch": epoch,
              "state_dim": state_dim, "action_dim": action_dim}
        if (epoch + 1) % 20 == 0 or epoch == args.epochs - 1:
            torch.save(ck, os.path.join(args.out_dir, "bc_latest.pt"))
        if epoch + 1 >= args.save_from and (epoch + 1) % 10 == 0:
            torch.save(ck, os.path.join(args.out_dir, f"bc_ep{epoch + 1}.pt"))


if __name__ == "__main__":
    main()
