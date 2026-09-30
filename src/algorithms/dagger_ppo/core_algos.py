"""PPO building blocks: GAE, advantage whitening, clipped surrogate, value and entropy losses."""
import torch


def gae(rewards, values, dones, last_value, gamma=0.99, lam=0.95):
    """Compute GAE(lambda) over [T, E] tensors; returns ``(advantages, returns)``."""
    T, E = rewards.shape
    advantages = torch.zeros_like(rewards)
    last = torch.zeros(E, device=rewards.device)
    for t in reversed(range(T)):
        nonterm = 1.0 - dones[t]
        nxt = last_value if t == T - 1 else values[t + 1]
        delta = rewards[t] + gamma * nxt * nonterm - values[t]
        last = delta + gamma * lam * nonterm * last
        advantages[t] = last
    return advantages, advantages + values


def whiten(x, eps=1e-8):
    """Normalize ``x`` to zero mean and unit standard deviation."""
    return (x - x.mean()) / (x.std() + eps)


def policy_loss(old_log_prob, log_prob, advantages, clip):
    """Clipped surrogate loss; also returns the k3 estimate of KL(old || new) for early stopping."""
    logratio = log_prob - old_log_prob
    ratio = logratio.exp()
    approx_kl = (ratio - 1 - logratio).mean()
    loss = torch.max(-advantages * ratio, -advantages * ratio.clamp(1 - clip, 1 + clip)).mean()
    return loss, approx_kl


def value_loss(values, returns):
    """Mean squared error between values and returns."""
    return (values - returns).pow(2).mean()


def entropy(dist):
    """Entropy summed over action dimensions, averaged over the batch."""
    return dist.entropy().sum(-1).mean()

