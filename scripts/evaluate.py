"""Evaluate the vision-based policy: success rate of opening the target door or drawer.

  python scripts/evaluate.py --ckpt runs/dagger_ppo/final.pt --split train val
"""
import argparse
import json
import os
import sys
from collections import defaultdict

from src.utils import exit_cleanly_on_sigterm, hard_exit, run_each  # noqa: E402

exit_cleanly_on_sigterm()
from src.envs import default_asset_root, list_assets, make_env  # noqa: E402  (before torch)
import numpy as np  # noqa: E402
import torch  # noqa: E402

from src.estimation import PRIOR_MODES, ArticulationFlow, load_prior  # noqa: E402
from src.algorithms.dagger_ppo.models import load_student  # noqa: E402


def main():
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--category", nargs="+", choices=["door", "drawer"], default=["door", "drawer"])
    ap.add_argument("--split", nargs="+", choices=["train", "val"], default=["train"])
    ap.add_argument("--chunk", type=int, default=40, help="assets per process")
    ap.add_argument("--ckpt", default="weights/dagger_ppo.pt", help="DAgger+PPO checkpoint")
    ap.add_argument("--bc_ckpt", default="weights/bc.pt", help="BC checkpoint the policy was trained from")
    ap.add_argument("--rounds", type=int, default=3)
    ap.add_argument("--steps", type=int, default=200)
    ap.add_argument("--seed", type=int, default=0)
    ap.add_argument("--asset_root", default=None)
    ap.add_argument("--prior", nargs="+", default=["weights/nap_door.pkl", "weights/nap_drawer.pkl"],
                    help="articulation prior file(s) (NAP samples)")
    ap.add_argument("--prior_mode", choices=PRIOR_MODES, default="densify",
                    help="densify: 16 best NAP directions, pivots from the observed part (released weights); "
                         "nap: every NAP sample is a particle (K=200)")
    ap.add_argument("--out_dir", default="results/released")
    ap.add_argument("--start", type=int, default=None, help=argparse.SUPPRESS)   # one chunk (internal)
    args = ap.parse_args()
    if args.start is None:
        run_chunks(args)
    else:
        run_chunk(args, args.category[0], args.split[0], args.start, args.start + args.chunk)


def chunk_file(args, category, split, start):
    return os.path.join(args.out_dir, f"{category}_{split}_{start}.json")


def summarize(paths):
    """Print the success rate per category and split from chunk result files."""
    groups = defaultdict(list)
    for path in paths:
        with open(path) as f:
            d = json.load(f)
        groups[(d["category"], d["split"])] += [r["success"] for r in d["records"]]
    for (category, split), succ in sorted(groups.items()):
        print(f"{category:6s} {split:9s} episodes={len(succ):5d}  success={sum(succ) / len(succ):.3f}")


def run_chunks(args):
    """One process per chunk; chunks with an existing result file are skipped."""
    root = args.asset_root or default_asset_root()
    cmds = []
    for category in args.category:
        for split in args.split:
            for start in range(0, len(list_assets(root, category, split)), args.chunk):
                if not os.path.exists(chunk_file(args, category, split, start)):
                    cmds.append([sys.executable, __file__, "--category", category, "--split", split,
                                 "--start", start, "--chunk", args.chunk, "--ckpt", args.ckpt,
                                 "--bc_ckpt", args.bc_ckpt, "--rounds", args.rounds, "--steps", args.steps,
                                 "--seed", args.seed, "--asset_root", root, "--prior", *args.prior,
                                 "--prior_mode", args.prior_mode, "--out_dir", args.out_dir])
    failed = run_each(cmds)
    files = [chunk_file(args, c, s, st) for c in args.category for s in args.split
             for st in range(0, len(list_assets(root, c, s)), args.chunk)]
    summarize([f for f in files if os.path.exists(f)])
    if failed:
        print(f"WARNING: {len(failed)} chunk(s) failed; the summary is incomplete, rerun to fill them in",
              file=sys.stderr)


def run_chunk(args, category, split, start, end):
    torch.manual_seed(args.seed); np.random.seed(args.seed)
    torch.set_grad_enabled(False)
    dev = "cuda:0"

    root = args.asset_root or default_asset_root()
    assets = list_assets(root, category, split)[start:end]
    if not assets:
        print(f"[eval] no assets in [{start}:{end})"); hard_exit(0)
    env = make_env("student", {"train": assets}, device_id=0, asset_root=root)
    student, a_mean, a_std = load_student(args.bc_ckpt, dev, args.ckpt)
    student.eval()
    perception = ArticulationFlow(env, load_prior(args.prior), dev, args.prior_mode)
    names = [p.name for p in env.selected_asset_path_list]

    records = []
    for r in range(args.rounds):
        obs = env.reset()
        pts = obs.points.float().to(dev)
        flow = perception.reset(pts[:, :, :3], pts[:, :, 6])
        success = torch.zeros(env.num_envs, dtype=torch.bool, device=dev)
        for _ in range(args.steps):
            pc = torch.cat([pts[:, :, :6], flow], dim=-1).contiguous()
            obs, _, _, info = env.step(student(pc, obs.obs.float().to(dev)) * a_std + a_mean)
            success |= info["successes"].to(dev).bool()
            pts = obs.points.float().to(dev)
            flow = perception.step(pts[:, :, :3], pts[:, :, 6])
        records += [{"asset": names[i // env.env_per_asset], "round": r, "success": bool(success[i])}
                    for i in range(env.num_envs)]
        print(f"[eval] {category}/{split} [{start}:{end}) round {r}: "
              f"success {success.float().mean():.3f}", flush=True)

    out = chunk_file(args, category, split, start)
    os.makedirs(args.out_dir, exist_ok=True)
    with open(out, "w") as f:
        json.dump({"category": category, "split": split, "start": start, "end": end,
                   "ckpt": args.ckpt, "records": records}, f)
    print(f"[eval] wrote {out}", flush=True)


if __name__ == "__main__":
    main()
    hard_exit(0)
