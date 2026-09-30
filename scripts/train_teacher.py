"""Train a state-based PPO teacher for one category.

  python scripts/train_teacher.py --category door
"""
import argparse
import os
import random

from src.utils import exit_cleanly_on_sigterm, hard_exit  # noqa: E402

exit_cleanly_on_sigterm()
from src.envs import default_asset_root, list_assets, make_env  # noqa: E402  (before torch)
import numpy as np  # noqa: E402
import torch  # noqa: E402

from src.algorithms.pregrasp_ppo.ppo import PPO  # noqa: E402

# Settings of the released teachers: environments per asset, train assets used, PPO target KL, iterations
DEFAULTS = {"door": dict(env_per_asset=1, num_train=None, desired_kl=0.005, max_iterations=40000),
            "drawer": dict(env_per_asset=5, num_train=200, desired_kl=0.01, max_iterations=2000)}


def main():
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--category", required=True, choices=["door", "drawer"])
    ap.add_argument("--log_dir", default=None, help="default: runs/teacher/<category>")
    ap.add_argument("--env_per_asset", type=int, default=None)
    ap.add_argument("--num_train", type=int, default=None, help="use the first N train assets")
    ap.add_argument("--desired_kl", type=float, default=None)
    ap.add_argument("--max_iterations", type=int, default=None)
    ap.add_argument("--eval_every", type=int, default=10)
    ap.add_argument("--resume", default="", help="continue from a model_<it>.tar")
    ap.add_argument("--seed", type=int, default=526)
    ap.add_argument("--asset_root", default=None)
    args = ap.parse_args()
    for k, v in DEFAULTS[args.category].items():
        if getattr(args, k) is None:
            setattr(args, k, v)
    args.log_dir = args.log_dir or f"runs/teacher/{args.category}"
    random.seed(args.seed); np.random.seed(args.seed); torch.manual_seed(args.seed); torch.cuda.manual_seed_all(args.seed)
    os.environ["PYTHONHASHSEED"] = str(args.seed)

    root = args.asset_root or default_asset_root()
    assets = {s: list_assets(root, args.category, s) for s in ("train", "val")}
    assets["train"] = assets["train"][:args.num_train]
    env = make_env("teacher", assets, device_id=0, asset_root=root, env_per_asset=args.env_per_asset)
    print(f"[teacher] {args.category}: {len(assets['train'])}/{len(assets['val'])} "
          f"train/val assets x {args.env_per_asset} envs", flush=True)
    ppo = PPO(env, args.log_dir, desired_kl=args.desired_kl, max_iterations=args.max_iterations,
              eval_every=args.eval_every)
    if args.resume:
        ppo.load(args.resume)
    ppo.train()


if __name__ == "__main__":
    main()
    hard_exit(0)
