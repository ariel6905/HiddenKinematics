"""Initial particles from the articulation prior (``densify`` or ``nap`` mode).

A prior file is a pickled dict ``{asset_name: entry}`` with B joint hypotheses per asset (e.g. NAP samples):
  dirs   [B, 3]  joint-axis direction, object frame
  score  [B]     lower = more likely (samples above INVALID_SCORE are treated as failed)
  piv    [B, 3]  pivot, normalized to the movable part's bounding box ([-1, 1] per axis); ``nap`` mode only
  typ    [B]     0 = prismatic, 1 = revolute; ``nap`` mode only
"""
import pickle

import numpy as np
import torch

PRISMATIC, REVOLUTE, RIGID = 0, 1, 2

SCORE_TEMP = 10.0        # Flattens the prior scores
N_DIR = 16               # Directions kept from the prior
M_PIV = 13               # Revolute pivot samples per direction
PIVOT_SCALE = 0.4        # Pivot scatter range, as a fraction of the observed part extent
INVALID_SCORE = 1e5      # "nap" mode: samples scored above this are failed NAP samples
RIGID_MASS = 0.05        # "nap" mode: prior mass of the rigid (fixed-joint) hypothesis
PRIOR_MODES = ("densify", "nap")


def load_prior(paths):
    """Merge prior files (e.g. one per category) into one ``{asset_name: entry}`` dict."""
    prior = {}
    for path in paths:
        with open(path, "rb") as f:
            prior.update(pickle.load(f))
    return prior


def quat_to_rot(q):
    """Convert an xyzw quaternion to a 3x3 rotation matrix (numpy)."""
    x, y, z, w = (float(v) for v in q[:4])
    return np.array([
        [1 - 2 * (y * y + z * z), 2 * (x * y - z * w), 2 * (x * z + y * w)],
        [2 * (x * y + z * w), 1 - 2 * (x * x + z * z), 2 * (y * z - x * w)],
        [2 * (x * z - y * w), 2 * (y * z + x * w), 1 - 2 * (x * x + y * y)]], dtype=float)


def sample_movable_points(pc, mask, n=256):
    """Sample ``n`` observed movable points per env; returns (points, valid)."""
    E = pc.shape[0]
    out = torch.zeros(E, n, 3, device=pc.device)
    valid = torch.zeros(E, dtype=torch.bool, device=pc.device)
    for e in range(E):
        idx = mask[e].nonzero(as_tuple=True)[0]
        if idx.numel() >= 8:
            out[e] = pc[e, idx[torch.randint(idx.numel(), (n,), device=pc.device)]]
            valid[e] = True
    return out.cpu().numpy(), valid.cpu().numpy()


def perp_extent(pts, center, u):
    """Size of the observed part in the plane perpendicular to ``u`` (where a hinge can lie)."""
    d = pts - center
    r = np.linalg.norm(d - np.outer(d @ u, u), axis=1)
    return float(np.quantile(r, 0.9) * 1.3 + 0.05)


def densify(rng, U, w, center, extent):
    """Top ``N_DIR`` directions -> ``M_PIV`` revolute + 1 prismatic particle each, plus one rigid particle."""
    Us, Ps, Ts, Ws = [], [], [], []
    for j in np.argsort(-w)[:N_DIR]:
        u = U[j]
        a = np.array([1., 0, 0]) if abs(u[0]) < 0.9 else np.array([0, 1., 0])
        e1 = np.cross(u, a)
        e1 /= np.linalg.norm(e1) + 1e-9
        e2 = np.cross(u, e1)
        for o in rng.uniform(-1, 1, (M_PIV, 2)) * (extent * PIVOT_SCALE):
            Us.append(u); Ps.append(center + o[0] * e1 + o[1] * e2); Ts.append(REVOLUTE); Ws.append(w[j])
        Us.append(u); Ps.append(center); Ts.append(PRISMATIC); Ws.append(w[j])
    Us.append(np.zeros(3)); Ps.append(np.zeros(3)); Ts.append(RIGID); Ws.append(0.05 * float(np.sum(Ws)))
    W = np.asarray(Ws, np.float32)
    W /= W.sum() + 1e-9
    return np.asarray(Us, np.float32), np.asarray(Ps, np.float32), np.asarray(Ts, np.int64), W


def nap_particles(entry, R, t, part_box):
    """Turn the valid NAP samples of one asset into world-frame particles plus a rigid one."""
    s = np.asarray(entry["score"], float)
    ok = s < INVALID_SCORE
    if not ok.any():
        raise ValueError("no valid NAP sample")
    U = np.asarray(entry["dirs"], float)[ok] @ R.T
    U /= np.linalg.norm(U, axis=1, keepdims=True) + 1e-8
    local = (np.asarray(part_box, float) - t) @ R                   # Box corners in the object frame
    lo, hi = local.min(0), local.max(0)
    P = (np.asarray(entry["piv"], float)[ok] * (hi - lo) / 2 + (hi + lo) / 2) @ R.T + t
    T = np.where(np.asarray(entry["typ"])[ok] == 0, PRISMATIC, REVOLUTE).astype(np.int64)
    s = s[ok]
    W = np.exp(-(s - s.min()) / (max(float(s.std()), 1e-3) * SCORE_TEMP))
    W = W / W.sum()
    U = np.concatenate([U, np.zeros((1, 3))])
    P = np.concatenate([P, np.zeros((1, 3))])
    T = np.concatenate([T, [RIGID]])
    W = np.concatenate([W * (1 - RIGID_MASS), [RIGID_MASS]])
    return U.astype(np.float32), P.astype(np.float32), T, W.astype(np.float32)


def initial_particles(prior, names, base_pos, base_quat, obs_pts, obs_valid, rng, mode="densify", part_boxes=None):
    """Build per-env particle sets (U, P, T, W) in the world frame."""
    if mode not in PRIOR_MODES:
        raise ValueError(f"unknown prior mode {mode!r}, expected one of {PRIOR_MODES}")
    out = []
    for e, name in enumerate(names):
        if name not in prior:
            raise KeyError(f"no articulation prior for asset {name}")
        entry = prior[name]
        R, t = quat_to_rot(base_quat[e]), base_pos[e]
        if mode == "nap":
            out.append(nap_particles(entry, R, t, part_boxes[e]))
            continue
        U = np.asarray(entry["dirs"], float) @ R.T
        U /= np.linalg.norm(U, axis=1, keepdims=True) + 1e-8
        s = np.asarray(entry["score"], float)
        w = np.exp(-(s - s.min()) / (max(float(s.std()), 1e-3) * SCORE_TEMP))
        if obs_valid[e]:
            center = obs_pts[e].mean(0)
            extent = perp_extent(obs_pts[e], center, U[int(np.argmax(w))])
        else:
            center, extent = t, 0.5
        out.append(densify(rng, U, w, center, extent))
    return out

