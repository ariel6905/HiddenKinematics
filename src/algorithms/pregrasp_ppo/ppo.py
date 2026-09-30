"""PPO for the state-based teacher on the 64-d privileged state (PartManip recipe)."""
import os
import time

import numpy as np
import torch
import torch.nn as nn
from torch.distributions import MultivariateNormal
from torch.utils.data.sampler import BatchSampler, SequentialSampler
from torch.utils.tensorboard import SummaryWriter


def mlp(sizes):
    """Linear layers with ELU in between (linear output)."""
    layers = []
    for i, (a, b) in enumerate(zip(sizes[:-1], sizes[1:])):
        layers.append(nn.Linear(a, b))
        if i < len(sizes) - 2:
            layers.append(nn.ELU())
    return nn.Sequential(*layers)


class ActorCritic(nn.Module):
    """Gaussian actor and critic MLPs with a state-independent ``log_std``."""

    def __init__(self, obs_dim, act_dim, hidden=(512, 512, 64), init_std=1.0):
        super().__init__()
        self.actor_mlp = mlp([obs_dim, *hidden, act_dim])
        self.critic_mlp = mlp([obs_dim, *hidden, 1])
        self.log_std = nn.Parameter(np.log(init_std) * torch.ones(act_dim))

    def _dist(self, mean):
        self.log_std.data = torch.clamp(self.log_std.data, -20, 20)
        # scale_tril holds exp(log_std)^2, so the action std is exp(2 * log_std) (as trained)
        return MultivariateNormal(mean, scale_tril=torch.diag(self.log_std.exp() * self.log_std.exp()))

    def act(self, obs):
        """Sample actions; returns ``(action, log_prob, value, mean)``."""
        mean = self.actor_mlp(obs)
        dist = self._dist(mean)
        a = dist.sample()
        return a, dist.log_prob(a), self.critic_mlp(obs), mean

    def evaluate(self, obs, actions):
        """Return ``(log_prob, entropy, value, mean)`` for ``actions``."""
        mean = self.actor_mlp(obs)
        dist = self._dist(mean)
        return dist.log_prob(actions), dist.entropy(), self.critic_mlp(obs), mean


class PPO:
    """PPO trainer for the "teacher" environment preset."""

    def __init__(self, env, log_dir, nsteps=5, epochs=8, minibatches=2, clip=0.1, gamma=0.99, lam=0.95,
                 value_coef=2.0, ent_coef=0.01, lr=3e-4, lr_bounds=(1e-7, 1e-3), desired_kl=0.005,
                 max_grad_norm=0.5, adam_eps=1e-5, max_iterations=20000, eval_every=10, eval_rounds=3):
        self.env, self.device = env, env.device
        self.n_train = env.env_num_train
        self.T = env.max_episode_length
        assert self.T % nsteps == 0
        self.nsteps, self.epochs, self.minibatches = nsteps, epochs, minibatches
        self.clip, self.gamma, self.lam = clip, gamma, lam
        self.value_coef, self.ent_coef, self.max_grad_norm = value_coef, ent_coef, max_grad_norm
        self.lr, self.lr_bounds, self.desired_kl = lr, lr_bounds, desired_kl
        self.max_iterations, self.eval_every, self.eval_rounds = max_iterations, eval_every, eval_rounds
        self.model = ActorCritic(env.state_dim, env.num_actions).to(self.device)
        self.optimizer = torch.optim.Adam(self.model.parameters(), lr=lr, eps=adam_eps)
        self.start_iteration, self.total_steps = 0, 0
        self.log_dir = log_dir
        os.makedirs(log_dir, exist_ok=True)
        self.writer = SummaryWriter(log_dir=log_dir, flush_secs=10)
        z = lambda *s: torch.zeros(nsteps, self.n_train, *s, device=self.device)
        self.buf = {"obs": z(env.state_dim), "act": z(env.num_actions), "rew": z(1), "done": z(1), "value": z(1),
                    "logp": z(1), "mu": z(env.num_actions), "sigma": z(env.num_actions)}

    # ------------------------------------------------------------------ Checkpoints
    def save(self, it):
        """Write ``model_<it>.tar`` to ``log_dir``."""
        torch.save({"iteration": it + 1, "model_state_dict": self.model.state_dict(),
                    "optimizer_state_dict": self.optimizer.state_dict(), "total_steps": self.total_steps},
                   os.path.join(self.log_dir, f"model_{it}.tar"))

    def load(self, path):
        """Resume from a ``model_<it>.tar`` checkpoint."""
        ck = torch.load(path, map_location=self.device)
        self.model.load_state_dict(ck["model_state_dict"])
        self.optimizer.load_state_dict(ck["optimizer_state_dict"])
        self.start_iteration, self.total_steps = ck["iteration"], ck["total_steps"]

    # ------------------------------------------------------------------ Evaluation
    @torch.no_grad()
    def evaluate(self, it):
        """Deterministic rollouts; logs and returns the success rate per split, averaged over rounds."""
        succ = torch.zeros(self.env.num_envs, self.eval_rounds, device=self.device)
        obs = self.env.reset()
        for r in range(self.eval_rounds):
            for _ in range(self.T):
                obs, _, _, info = self.env.step(self.env.uncanonicalize(self.model.actor_mlp(obs.state)))
                succ[:, r] = torch.logical_or(info["successes"].to(self.device).bool(), succ[:, r].bool())
        a = self.n_train
        res = {"train": succ[:a].mean().item(), "val": succ[a:].mean().item()}
        for k, v in res.items():
            self.writer.add_scalar(f"Test/TestSuccessRate/{k}", v, it)
        print(f"[teacher] it {it} eval success " + " ".join(f"{k} {v:.3f}" for k, v in res.items()), flush=True)
        return res

    # ------------------------------------------------------------------ Training
    def train(self):
        """Train until ``max_iterations``, evaluating and saving every ``eval_every`` iterations."""
        self.evaluate(self.start_iteration)
        n = self.n_train
        for it in range(self.start_iteration, self.max_iterations):
            obs = self.env.reset()
            if it % self.eval_every == 0:
                self.evaluate(it)
                self.save(it)
            start = time.time()
            ep_rew, succ = [], torch.zeros(self.env.num_envs, device=self.device)
            for i in range(self.T):
                with torch.no_grad():
                    a, logp, value, mu = self.model.act(obs.state[:n])
                    actions = a
                    if self.env.num_envs > n:                                 # Validation envs: deterministic
                        actions = torch.cat((a, self.model.actor_mlp(obs.state[n:])))
                    next_obs, rew, done, info = self.env.step(self.env.uncanonicalize(actions))
                step = i % self.nsteps
                for k, v in (("obs", obs.state[:n]), ("act", a), ("rew", rew[:n].view(-1, 1)), ("done", done[:n].view(-1, 1)),
                             ("value", value), ("logp", logp.view(-1, 1)), ("mu", mu),
                             ("sigma", self.model.log_std.repeat(n, 1).detach())):
                    self.buf[k][step].copy_(v)
                obs = next_obs
                ep_rew.append(rew[:n].mean().item())
                succ = torch.logical_or(succ.bool(), info["successes"].to(self.device).bool()).float()
                if (i + 1) % self.nsteps == 0:
                    with torch.no_grad():
                        last_value = self.model.critic_mlp(obs.state[:n])
                    stats = self.update(it, last_value)
            self.total_steps += self.T * self.env.num_envs
            self.writer.add_scalar("Train/mean_reward", float(np.mean(ep_rew)), it)
            self.writer.add_scalar("Train/success_rate", succ[:n].mean().item(), it)
            self.writer.add_scalar("Train/lr", self.lr, it)
            print(f"[teacher] it {it} reward {np.mean(ep_rew):.3f} success {succ[:n].mean().item():.3f} "
                  f"value_loss {stats[0]:.4f} surrogate {stats[1]:.4f} lr {self.lr:.2e} "
                  f"time {time.time() - start:.1f}s", flush=True)
        self.save(self.max_iterations)

    def update(self, it, last_value):
        """GAE and PPO epochs on the rollout buffer; returns the mean value and surrogate losses."""
        b = self.buf
        adv, ret = torch.zeros_like(b["value"]), torch.zeros_like(b["value"])
        running = 0
        for t in reversed(range(self.nsteps)):
            nxt = last_value if t == self.nsteps - 1 else b["value"][t + 1]
            nonterminal = 1.0 - b["done"][t]
            delta = b["rew"][t] + nonterminal * self.gamma * nxt - b["value"][t]
            running = delta + nonterminal * self.gamma * self.lam * running
            ret[t] = running + b["value"][t]
        adv = ret - b["value"]
        flat = lambda x: x.view(-1, x.shape[-1])
        obs, act, old_logp, old_mu, old_sigma = flat(b["obs"]), flat(b["act"]), flat(b["logp"]), flat(b["mu"]), flat(b["sigma"])
        ret_f, adv_f = flat(ret), flat(adv)
        size = self.nsteps * self.n_train
        v_losses, s_losses = [], []
        for _ in range(self.epochs):
            for idx in BatchSampler(SequentialSampler(range(size)), max(1, size // self.minibatches), drop_last=True):
                logp, ent, value, mu = self.model.evaluate(obs[idx], act[idx])
                sigma = self.model.log_std.repeat(len(idx), 1)
                a = adv_f[idx]
                a = (a - a.mean()) / (a.std() + 1e-8)
                # Adaptive learning rate: keep the KL to the rollout policy near desired_kl
                kl = torch.sum(sigma - old_sigma[idx] + (torch.square(old_sigma[idx].exp()) + torch.square(old_mu[idx] - mu))
                               / (2.0 * torch.square(sigma.exp())) - 0.5, axis=-1).mean()
                if kl > self.desired_kl * 2.0:
                    self.lr = max(self.lr_bounds[0], self.lr / 1.5)
                elif self.desired_kl / 2.0 > kl > 0.0:
                    self.lr = min(self.lr_bounds[1], self.lr * 1.5)
                for g in self.optimizer.param_groups:
                    g["lr"] = self.lr
                ratio = torch.exp(logp - torch.squeeze(old_logp[idx]))
                surrogate = torch.max(-torch.squeeze(a) * ratio,
                                      -torch.squeeze(a) * torch.clamp(ratio, 1.0 - self.clip, 1.0 + self.clip)).mean()
                value_loss = (ret_f[idx] - value).pow(2).mean()
                loss = surrogate + self.value_coef * value_loss - self.ent_coef * ent.mean()
                self.optimizer.zero_grad()
                loss.backward()
                nn.utils.clip_grad_norm_(self.model.parameters(), self.max_grad_norm)
                self.optimizer.step()
                v_losses.append(value_loss.item()); s_losses.append(surrogate.item())
        for g in self.optimizer.param_groups:                                  # Linear decay (overwritten by the adaptive rate)
            g["lr"] = self.lr * (1 - it / self.max_iterations)
        return float(np.mean(v_losses)), float(np.mean(s_losses))
