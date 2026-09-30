"""DAgger + PPO fine-tuning of the vision-based policy (three passes over the training shards).

  python scripts/train_dagger_ppo.py --bc_ckpt <bc.pt> --teacher_door <door.tar> --teacher_drawer <drawer.tar>
"""
import argparse
import os
import random
import shutil
import sys
from collections import deque

from src.utils import (average_parameters, average_scalar, barrier, exit_cleanly_on_sigterm, hard_exit,
                       init_distributed, run_each)  # noqa: E402

_on_exit = exit_cleanly_on_sigterm()
from src.envs import list_assets, make_env, read_shard, default_asset_root  # noqa: E402  (before torch)
import numpy as np  # noqa: E402
import torch  # noqa: E402
import torch.nn as nn  # noqa: E402
import torch.nn.functional as F  # noqa: E402
from torch.distributions import Normal  # noqa: E402

from src.estimation import PRIOR_MODES, ArticulationFlow, load_prior  # noqa: E402
from src.algorithms.dagger_ppo.core_algos import entropy, gae, policy_loss, value_loss, whiten  # noqa: E402
from src.algorithms.dagger_ppo.models import Critic, load_student  # noqa: E402
from src.algorithms.pregrasp_ppo import DoorDrawerTeacher  # noqa: E402


def parse_args():
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--shards", default="configs/shards.yaml", help="training shards of the train split")
    ap.add_argument("--shard", type=int, default=None, help="train one segment on this shard (default: full schedule)")
    ap.add_argument("--passes", type=int, default=3, help="full schedule: passes over the shards")
    ap.add_argument("--launch", default="torchrun --standalone --nproc_per_node 3",
                    help="full schedule: how to start one process per GPU for a segment")
    ap.add_argument("--asset_root", default=None)
    ap.add_argument("--prior", nargs="+", default=["weights/nap_door.pkl", "weights/nap_drawer.pkl"],
                    help="articulation prior file(s) (NAP samples)")
    ap.add_argument("--prior_mode", choices=PRIOR_MODES, default="densify",
                    help="densify: 16 best NAP directions, pivots from the observed part (released weights); "
                         "nap: every NAP sample is a particle (K=200)")
    ap.add_argument("--teacher_door", default="weights/teacher_door.tar")
    ap.add_argument("--teacher_drawer", default="weights/teacher_drawer.tar")
    ap.add_argument("--bc_ckpt", default="weights/bc.pt", help="BC checkpoint: architecture, action normalization, init weights")
    ap.add_argument("--init_ckpt", default="", help="continue from this DAgger+PPO checkpoint instead of BC")
    ap.add_argument("--save_dir", default="runs/dagger_ppo")
    ap.add_argument("--run_name", default="", help="segment name (set by the full schedule)")
    ap.add_argument("--iters", type=int, default=50)
    ap.add_argument("--start_iter", type=int, default=0,
                    help="global iteration of this segment's first iteration (DAgger-weight schedule, --dagger_mode joint)")
    ap.add_argument("--episode_len", type=int, default=200)
    ap.add_argument("--seed", type=int, default=0)
    # DAgger
    ap.add_argument("--dagger_mode", choices=("separate", "joint"), default="separate",
                    help="separate: own Adam for DAgger, so the weight decay has no effect (released weights); "
                         "joint: one optimizer on w_dag * DAgger + PPO, the weight decays to --dag_floor")
    ap.add_argument("--dagger_lr", type=float, default=3e-4, help="--dagger_mode separate only")
    ap.add_argument("--dagger_batch", type=int, default=32)
    ap.add_argument("--dagger_decay", type=float, default=0.05,
                    help="DAgger weight = max(dag_floor, 1 - it * decay); it counts from --start_iter in joint mode")
    ap.add_argument("--dag_floor", type=float, default=0.05)
    ap.add_argument("--buf_iters", type=int, default=6)
    ap.add_argument("--beta", type=float, default=0.5, help="P(execute teacher action) during critic warm-up")
    # PPO
    ap.add_argument("--lr", type=float, default=1e-4)
    ap.add_argument("--critic_lr", type=float, default=3e-4)
    ap.add_argument("--critic_warmup", type=int, default=5, help="iterations that train only the critic")
    ap.add_argument("--clip", type=float, default=0.02)
    ap.add_argument("--target_kl", type=float, default=0.12, help="skip PPO minibatches whose KL exceeds this")
    ap.add_argument("--ent_coef", type=float, default=0.01)
    ap.add_argument("--init_log_std", type=float, default=-1.0)
    ap.add_argument("--ppo_epochs", type=int, default=4)
    ap.add_argument("--ppo_batch", type=int, default=256)
    ap.add_argument("--gamma", type=float, default=0.99)
    ap.add_argument("--lam", type=float, default=0.95)
    ap.add_argument("--vf_coef", type=float, default=0.5)
    ap.add_argument("--max_grad_norm", type=float, default=0.5)
    ap.add_argument("--info_coef", type=float, default=0.1, help="weight of the belief-entropy-decrease reward")
    # evaluation during training
    ap.add_argument("--eval_every", type=int, default=25, help="also evaluated after the last iteration")
    ap.add_argument("--eval_rounds", type=int, default=1)
    return ap, ap.parse_args()


def run_schedule(ap, args):
    """Run the segments one after another, each as its own (multi-GPU) launch."""
    os.environ["CUDA_DEVICE_ORDER"] = "PCI_BUS_ID"   # CUDA and Vulkan (cameras) must enumerate GPUs identically
    schedule = {"shard", "passes", "launch", "run_name", "start_iter", "init_ckpt"}
    common = []
    for a in ap._actions:
        v = getattr(args, a.dest, None)
        if a.option_strings and a.dest not in schedule and a.dest != "help" and v not in (None, ""):
            common += [a.option_strings[0], *(v if isinstance(v, list) else [v])]
    prev, seg = args.init_ckpt, 0
    for p in range(1, args.passes + 1):
        for shard in range(3):
            name = f"pass{p}_shard{shard}"
            cmd = [*args.launch.split(), __file__, *common, "--shard", shard, "--run_name", name,
                   "--start_iter", seg * args.iters] + (["--init_ckpt", prev] if prev else [])
            if run_each([cmd]):
                sys.exit(f"segment {name} failed; rerun it with --init_ckpt {prev}" if prev else f"segment {name} failed")
            prev, seg = os.path.join(args.save_dir, f"{name}_latest.pt"), seg + 1
    shutil.copy(os.path.join(args.save_dir, f"{name}_best.pt"), os.path.join(args.save_dir, "final.pt"))
    print(f"final policy: {args.save_dir}/final.pt (best evaluation of the last segment); last iterate: {prev}")


def point_cloud(obs, flow, device):
    """Policy input [E, N, 9]: xyz + rgb + articulation flow."""
    return torch.cat([obs.points[:, :, :6].float().to(device), flow], dim=-1).contiguous()


@torch.no_grad()
def evaluate(env, student, perception, a_mean, a_std, device, steps, rounds):
    """Deterministic rollouts; returns the (all, door, drawer) success rates on this rank's assets."""
    succ = torch.zeros(env.num_envs, device=device)
    for _ in range(rounds):
        obs = env.reset()
        flow = perception.reset(obs.points[:, :, :3].float().to(device))
        for _ in range(steps):
            pc = point_cloud(obs, flow, device)
            obs, _, _, info = env.step(student(pc, obs.obs.float().to(device)) * a_std + a_mean)
            succ = torch.logical_or(succ.bool(), info["successes"].to(device).bool()).float()
            flow = perception.step(obs.points[:, :, :3].float().to(device))
    d = env.is_door
    return succ.mean().item(), succ[d].mean().item(), succ[~d].mean().item()


def main():
    ap, args = parse_args()
    if args.shard is None:
        run_schedule(ap, args)
        return
    rank, world, local = init_distributed()
    device = f"cuda:{local}"
    seed = args.seed + rank
    np.random.seed(seed); torch.manual_seed(seed); torch.cuda.manual_seed_all(seed); random.seed(seed)
    os.makedirs(args.save_dir, exist_ok=True)

    root = args.asset_root or default_asset_root()
    doors = read_shard(args.shards, "door", args.shard)[rank::world]            # This rank's share of the assets
    drawers = read_shard(args.shards, "drawer", args.shard)[rank::world]
    train = list_assets(root, "door", "train", doors) + list_assets(root, "drawer", "train", drawers)
    env = make_env("student", {"train": train}, device_id=local, asset_root=root)
    E = env.num_envs
    teacher = DoorDrawerTeacher(env, args.teacher_door, args.teacher_drawer, device)
    perception = ArticulationFlow(env, load_prior(args.prior), device, args.prior_mode)

    student, a_mean, a_std = load_student(args.bc_ckpt, device)
    state_dim, act_dim = student.head[0].in_features - student.backbone.out_dim, student.head[-1].out_features
    for p in student.backbone.parameters():                            # Frozen encoder, BatchNorm in eval mode
        p.requires_grad_(False)
    student.backbone.eval()
    log_std = nn.Parameter(torch.full((act_dim,), float(args.init_log_std), device=device))
    if args.init_ckpt:
        ck = torch.load(args.init_ckpt, map_location="cpu")
        student.load_state_dict(ck["policy"])
        student.backbone.eval()
        with torch.no_grad():
            log_std.copy_(ck["log_std"].to(device).float().reshape(-1))
    policy_params = list(student.head.parameters()) + [log_std]
    critic = Critic(state_dim).to(device)
    opt = torch.optim.Adam(policy_params, lr=args.lr)
    opt_dagger = torch.optim.Adam(policy_params, lr=args.dagger_lr)      # Own Adam state (--dagger_mode separate)
    opt_critic = torch.optim.Adam(critic.parameters(), lr=args.critic_lr)
    replay = deque(maxlen=args.buf_iters)                               # (feature, state, teacher action)

    def head(feat, state):
        return student.head(torch.cat([state, feat], dim=-1))

    def joint_update(bi, w_dag, warmup, ret, adv, Ap, LPp, F_, S_, dag_losses, pg_losses, kls, v_losses):
        """``--dagger_mode joint``: one critic step, then one policy step on ``w_dag * DAgger + PPO``."""
        feat, st = F_[bi].to(device), S_[bi].to(device)
        vl = value_loss(critic(st), ret[bi].to(device))
        opt_critic.zero_grad(); (args.vf_coef * vl).backward()
        nn.utils.clip_grad_norm_(critic.parameters(), args.max_grad_norm); opt_critic.step()
        v_losses.append(vl.item())
        buf = replay[np.random.randint(len(replay))]
        di = torch.randint(buf[0].shape[0], (args.dagger_batch,))
        target = (buf[2][di].to(device) - a_mean) / a_std
        dag = F.mse_loss(head(buf[0][di].to(device), buf[1][di].to(device)), target)
        loss = w_dag * dag
        dag_losses.append(loss.item())
        if not warmup:                                  # PPO term, dropped beyond the soft KL stop
            mean = head(feat, st)
            dist = Normal(mean, log_std.exp().expand_as(mean))
            pg, kl = policy_loss(LPp[bi].to(device), dist.log_prob(Ap[bi].to(device)).sum(-1), adv[bi].to(device), args.clip)
            kls.append(kl.item())
            if kl.item() <= args.target_kl:
                loss = loss + pg - args.ent_coef * entropy(dist)
                pg_losses.append(pg.item())
        opt.zero_grad(); loss.backward()
        nn.utils.clip_grad_norm_(policy_params, args.max_grad_norm); opt.step()

    def save(name, **extra):
        if rank == 0:
            torch.save({"policy": student.state_dict(), "log_std": log_std.detach().cpu(),
                        "a_mean": a_mean.cpu(), "a_std": a_std.cpu(), "state_dim": state_dim, **extra},
                       os.path.join(args.save_dir, f"{args.run_name}_{name}.pt"))

    _on_exit[0] = lambda: save("latest")
    save("latest")
    print(f"[train] {args.run_name} rank {rank}/{world}: {E} envs "
          f"({int(env.is_door.sum())} doors + {int((~env.is_door).sum())} drawers)", flush=True)
    best, sd, sdr = (average_scalar(x, world, device) for x in
                     evaluate(env, student, perception, a_mean, a_std, device, args.episode_len, args.eval_rounds))
    if rank == 0:
        print(f"[train] initial success {best:.3f} (door {sd:.3f}, drawer {sdr:.3f})", flush=True)
    save("best", it=-1, succ=best)          # Starting policy; replaced by any better evaluation

    info_mean = info_std = None
    T = args.episode_len
    for it in range(args.iters):
        warmup = it < args.critic_warmup
        # ---------------------------------------------------------------- Rollout
        feats, states, labels, acts, logps, values, rewards, dones = [], [], [], [], [], [], [], []
        reward_terms = {}
        ep_succ = torch.zeros(E, device=device)
        obs = env.reset()
        flow = perception.reset(obs.points[:, :, :3].float().to(device))
        for t in range(T):
            pc = point_cloud(obs, flow, device)
            st = obs.obs.float().to(device)
            with torch.no_grad():
                feat = student.backbone(pc)
                mean = head(feat, st)
                label = teacher(obs)
                val = critic(st)
                if not warmup:
                    dist = Normal(mean, log_std.exp().expand_as(mean))
                    a = dist.sample()
                    lp = dist.log_prob(a).sum(-1)
                    action = a * a_std + a_mean
                else:                               # Critic warm-up: deterministic student / teacher mix
                    a, lp = mean, torch.zeros(mean.shape[0], device=device)
                    action = label if float(torch.rand(())) < args.beta else mean * a_std + a_mean
            feats.append(feat.cpu()); states.append(st.cpu()); labels.append(label.cpu())
            acts.append(a.cpu()); logps.append(lp.cpu()); values.append(val.cpu())
            obs, rew, done, info = env.step(action)
            for k, v in env.reward_terms.items():
                reward_terms[k] = reward_terms.get(k, 0.0) + float(v.float().mean())
            rewards.append(rew.detach().float().cpu()); dones.append(done.detach().float().cpu())
            ep_succ = torch.logical_or(ep_succ.bool(), info["successes"].to(device).bool()).float()
            flow = perception.step(obs.points[:, :, :3].float().to(device))
            # Information-gain reward: belief-entropy decrease, normalized by running statistics
            gain = perception.estimator.information_gain().float()
            m, s = gain.mean(), gain.std().clamp_min(1e-3)
            info_mean = m if info_mean is None else 0.99 * info_mean + 0.01 * m
            info_std = s if info_std is None else 0.99 * info_std + 0.01 * s
            gain = ((gain - info_mean) / (info_std + 1e-9)).clamp(-5, 5)
            rewards[-1] = rewards[-1] + (args.info_coef * gain).detach().float().cpu()
            reward_terms["info"] = reward_terms.get("info", 0.0) + float(gain.mean())

        # ---------------------------------------------------------------- DAgger replay
        F_, S_, L_ = (torch.stack(x).reshape(T * E, -1) for x in (feats, states, labels))
        replay.append((F_, S_, L_))
        n_replay = sum(r[0].shape[0] for r in replay)

        # ---------------------------------------------------------------- Update
        w_it = it + (args.start_iter if args.dagger_mode == "joint" else 0)
        w_dag = max(args.dag_floor, 1.0 - w_it * args.dagger_decay)
        with torch.no_grad():
            last_val = critic(obs.obs.float().to(device))
        R, V, D = torch.stack(rewards).to(device), torch.stack(values).to(device), torch.stack(dones).to(device)
        adv, ret = gae(R, V, D, last_val, args.gamma, args.lam)
        adv, ret = whiten(adv.reshape(-1)), ret.reshape(-1)
        ev = float(1.0 - (ret - V.reshape(-1)).var() / (ret.var() + 1e-8))
        Ap, LPp = torch.stack(acts).reshape(T * E, -1), torch.stack(logps).reshape(-1)
        N = T * E
        dag_losses, pg_losses, kls, v_losses = [], [], [], []
        joint = args.dagger_mode == "joint"
        for _ in range(args.ppo_epochs):
            perm = torch.randperm(N)
            for s0 in range(0, N, args.ppo_batch):
                if joint:
                    joint_update(perm[s0:s0 + args.ppo_batch], w_dag, warmup, ret, adv, Ap, LPp, F_, S_,
                                 dag_losses, pg_losses, kls, v_losses)
                    continue
                buf = replay[np.random.randint(len(replay))]            # DAgger step
                di = torch.randint(buf[0].shape[0], (args.dagger_batch,))
                target = (buf[2][di].to(device) - a_mean) / a_std
                loss = F.mse_loss(head(buf[0][di].to(device), buf[1][di].to(device)), target) * w_dag
                opt_dagger.zero_grad(); loss.backward()
                nn.utils.clip_grad_norm_(policy_params, args.max_grad_norm); opt_dagger.step()
                dag_losses.append(loss.item())
                bi = perm[s0:s0 + args.ppo_batch]                        # Critic step
                feat, st = F_[bi].to(device), S_[bi].to(device)
                vl = value_loss(critic(st), ret[bi].to(device))
                opt_critic.zero_grad(); (args.vf_coef * vl).backward()
                nn.utils.clip_grad_norm_(critic.parameters(), args.max_grad_norm); opt_critic.step()
                v_losses.append(vl.item())
                if warmup:
                    continue
                mean = head(feat, st)                                    # PPO step
                dist = Normal(mean, log_std.exp().expand_as(mean))
                pg, kl = policy_loss(LPp[bi].to(device), dist.log_prob(Ap[bi].to(device)).sum(-1), adv[bi].to(device), args.clip)
                kls.append(kl.item())
                if kl.item() > args.target_kl:
                    continue
                loss = pg - args.ent_coef * entropy(dist)
                opt.zero_grad(); loss.backward()
                nn.utils.clip_grad_norm_(policy_params, args.max_grad_norm); opt.step()
                pg_losses.append(pg.item())
        average_parameters(list(student.head.parameters()) + [log_std] + list(critic.parameters()), world)

        msg = (f"[train] it={it} succ={int(ep_succ.sum())}/{E} replay={n_replay} w_dag={w_dag:.3f}"
               f" dagger_mse={np.mean(dag_losses):.4f}"
               f" | reward[" + " ".join(f"{k}={v / T:.3f}" for k, v in reward_terms.items()) + "]"
               f" | ppo_pg={np.mean(pg_losses) if pg_losses else 0:.4f} kl={np.mean(kls) if kls else 0:.4f}"
               f" steps={len(pg_losses)}{' (warmup)' if warmup else ''} | critic EV={ev:.3f} loss={np.mean(v_losses):.4f}")
        if it % args.eval_every == 0 or it == args.iters - 1:
            succ, sd, sdr = (average_scalar(x, world, device) for x in
                             evaluate(env, student, perception, a_mean, a_std, device, args.episode_len, args.eval_rounds))
            msg += f" | eval success {succ:.3f} (door {sd:.3f}, drawer {sdr:.3f})"
            if succ > best:
                best = succ
                save("best", it=it, succ=succ)
                msg += " (best)"
        if rank == 0:
            print(msg, flush=True)
        save("latest", it=it)
    barrier(world)


if __name__ == "__main__":
    main()
    hard_exit(0)
