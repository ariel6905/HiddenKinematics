"""Collect behavior-cloning demonstrations with the state teachers.

  python scripts/collect_demos.py --teacher_door <door.tar> --teacher_drawer <drawer.tar>
"""
import argparse
import os
import sys

from src.utils import exit_cleanly_on_sigterm, hard_exit, run_each  # noqa: E402

exit_cleanly_on_sigterm()
from src.envs import default_asset_root, list_assets, make_env  # noqa: E402  (before torch)
import numpy as np  # noqa: E402
import torch  # noqa: E402

from src.estimation import PRIOR_MODES, ArticulationFlow, load_prior  # noqa: E402
from src.algorithms.pregrasp_ppo import DoorDrawerTeacher  # noqa: E402


def main():
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--category", nargs="+", choices=["door", "drawer"], default=["door", "drawer"])
    ap.add_argument("--chunk", type=int, default=30, help="assets per process")
    ap.add_argument("--teacher_door", default="weights/teacher_door.tar")
    ap.add_argument("--teacher_drawer", default="weights/teacher_drawer.tar")
    ap.add_argument("--save_dir", default="data/demos")
    ap.add_argument("--traj_per_asset", type=int, default=3)
    ap.add_argument("--episode_len", type=int, default=200)
    ap.add_argument("--seed", type=int, default=0)
    ap.add_argument("--asset_root", default=None)
    ap.add_argument("--prior", nargs="+", default=["weights/nap_door.pkl", "weights/nap_drawer.pkl"],
                    help="articulation prior file(s) (NAP samples)")
    ap.add_argument("--prior_mode", choices=PRIOR_MODES, default="densify",
                    help="densify: 16 best NAP directions, pivots from the observed part (released weights); "
                         "nap: every NAP sample is a particle (K=200)")
    ap.add_argument("--start", type=int, default=None, help=argparse.SUPPRESS)   # one chunk (internal)
    args = ap.parse_args()
    root = args.asset_root or default_asset_root()
    if args.start is None:
        failed = run_each([[sys.executable, __file__, "--category", category, "--start", start,
                            "--chunk", args.chunk, "--teacher_door", args.teacher_door,
                            "--teacher_drawer", args.teacher_drawer, "--save_dir", args.save_dir,
                            "--traj_per_asset", args.traj_per_asset, "--episode_len", args.episode_len,
                            "--seed", args.seed, "--asset_root", root, "--prior", *args.prior,
                            "--prior_mode", args.prior_mode]
                           for category in args.category
                           for start in range(0, len(list_assets(root, category, "train")), args.chunk)])
        if failed:
            print(f"WARNING: {len(failed)} chunk(s) failed", file=sys.stderr)
        return
    args.category, args.end = args.category[0], args.start + args.chunk
    torch.manual_seed(args.seed); np.random.seed(args.seed)
    dev = "cuda:0"

    assets = list_assets(root, args.category, "train")[args.start:args.end]
    if not assets:
        print(f"[demos] no assets in [{args.start}:{args.end})"); hard_exit(0)
    env = make_env("student", {"train": assets}, device_id=0, asset_root=root)
    teacher = DoorDrawerTeacher(env, args.teacher_door, args.teacher_drawer, dev)
    prior = load_prior(args.prior)
    names = [p.name for p in env.selected_asset_path_list]
    E = env.num_envs
    store = {n: {k: [] for k in ("observations", "actions", "pcs", "success")} for n in names}

    for traj in range(args.traj_per_asset):
        perception = ArticulationFlow(env, prior, dev, args.prior_mode)
        obs = env.reset()
        pts = obs.points.float().to(dev)
        flow = perception.reset(pts[:, :, :3], pts[:, :, 6])
        buf = {k: [] for k in ("observations", "actions", "pcs")}
        success = torch.zeros(E, dtype=torch.bool, device=dev)
        for t in range(args.episode_len):
            action = teacher(obs)
            buf["observations"].append(obs.obs.cpu().numpy())
            buf["actions"].append(action.cpu().numpy())
            buf["pcs"].append(torch.cat([pts[:, :, :6], flow, pts[:, :, 6:7]], dim=-1).cpu().numpy().astype(np.float32))
            obs, _, _, info = env.step(action)
            success |= info["successes"].to(dev).bool()
            pts = obs.points.float().to(dev)
            flow = perception.step(pts[:, :, :3], pts[:, :, 6])
        stacked = {k: np.stack(v, 0) for k, v in buf.items()}                  # [T, E, ...]
        for e, name in enumerate(names):
            for k in buf:
                store[name][k].append(stacked[k][:, e])
            store[name]["success"].append(float(success[e]))
        print(f"[collect] {args.category} [{args.start}:{args.end}) trajectory {traj + 1}/{args.traj_per_asset}: "
              f"teacher success {success.float().mean():.3f}", flush=True)

    out_dir = os.path.join(args.save_dir, args.category)
    os.makedirs(out_dir, exist_ok=True)
    for name, d in store.items():
        np.save(os.path.join(out_dir, name), {
            "observations": np.stack(d["observations"], 0), "actions": np.stack(d["actions"], 0),
            "pcs": np.stack(d["pcs"], 0), "success": np.asarray(d["success"], np.float32)})
    print(f"[collect] wrote {len(store)} assets to {out_dir}", flush=True)


if __name__ == "__main__":
    main()
    hard_exit(0)
