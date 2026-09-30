"""Online articulation estimation with a particle filter, rendered as the articulation flow."""
import math

import numpy as np
import torch

from .icp import icp, screw_axis
from .particles import ParticleSet
from .prior import PRISMATIC, REVOLUTE, initial_particles, sample_movable_points
from .segmentation import sample_points

# -- Tracking / likelihood --------------------------------------------------------------------
N_TRACK = 512            # Tracked movable points (first-frame sample, moved by accumulated ICP)
ICP_ITERS, ICP_TRIM, ICP_REG = 8, 0.1, 0.3
SIGMA = 0.03             # Centroid-displacement likelihood std (m)
MIN_DISP = 0.06          # Belief is only updated once the centroid moved this far (m)
COV_WEIGHT = 1.0         # Weight of the second-moment (shape rotation) residual for revolute
# -- Candidate proposals ----------------------------------------------------------------------
MOTION_SEEDS = 5         # Prismatic candidates around the observed displacement direction
MOTION_FAN_DEG = 8.0     # Half-angle of that fan (deg)
RESEED_GROWTH = 1.5      # Propose again once the displacement grew by this factor
SCREW_MIN_DEG = 5.0      # Screw (revolute) candidate once the part rotated this much (deg)
# -- Resampling -------------------------------------------------------------------------------
ESS_FRAC = 0.5           # Resample when ESS < ESS_FRAC * #valid particles
JITTER_AXIS, JITTER_PIVOT = 0.05, 0.03
JITTER_DECAY, JITTER_FLOOR = 0.85, 0.25
# -- Flow rendering ---------------------------------------------------------------------------
FLOW_BETA = 0.25         # Radial-gradient compression: length m -> 1 + beta * (m - 1)
FLOW_MAX_NORM = 3.0


class ArticulationEstimator:
    """Particle-filter articulation belief per env and the articulation-flow renderer."""

    def __init__(self, task, prior, device, prior_mode="densify"):
        self.task, self.prior, self.device = task, prior, device
        self.prior_mode = prior_mode              # How the prior becomes particles, see prior.py
        self.E = task.env_num
        self.env_per_asset = task.env_per_asset
        self.names = [p.name for p in task.selected_asset_path_list]
        self.pf = ParticleSet(self.E, device)
        self.rng = np.random.default_rng(0)       # Pivot scatter of the prior (persists across resets)
        self._ent_prev = None

    # ------------------------------------------------------------------ Belief update
    def reset(self, xyz, movable, given_seg=None):
        """Initialize the belief and the tracker from the first frame."""
        xyz = xyz.to(self.device).float()
        movable = movable.to(self.device)
        obs_pts, obs_valid = sample_movable_points(xyz, movable)
        root = self.task.cabinet_root_tensor.detach().cpu().numpy()
        boxes = self.task.init_part_bbox_tensor.detach().cpu().numpy() if self.prior_mode == "nap" else None
        self.pf.reset(initial_particles(self.prior, [self.names[e // self.env_per_asset] for e in range(self.E)],
                                        root[:, :3], root[:, 3:7], obs_pts, obs_valid, self.rng,
                                        mode=self.prior_mode, part_boxes=boxes))
        pts0, _ = sample_points(xyz, movable, N_TRACK)
        self._pts0 = pts0
        self._cent0 = pts0.mean(1)
        X = pts0 - self._cent0.unsqueeze(1)
        self._cov0 = X.transpose(1, 2) @ X / pts0.shape[1]
        self._R = None                     # Accumulated rigid motion of the tracked points
        self._t = None
        self._seeded = None                # Prismatic motion candidates proposed?
        self._seed_disp = None             # Displacement at the last proposal
        self._screw_slot = -torch.ones(self.E, dtype=torch.long, device=self.device)
        self._jitter = None
        self._xyz0, self._given0 = xyz, (None if given_seg is None else given_seg.to(self.device))
        self._xyz, self._given = xyz, self._given0

    def update(self, xyz, movable, given_seg=None):
        """Track the movable part in a new frame and update the belief."""
        xyz = xyz.to(self.device).float()
        self._xyz, self._given = xyz, (None if given_seg is None else given_seg.to(self.device))
        tracked = self._track(xyz, movable.to(self.device))
        self._reweight(tracked)

    def _track(self, xyz, movable):
        """Move the first-frame points by the accumulated ICP motion (no frame-to-frame drift)."""
        cur, valid = sample_points(xyz, movable, 4 * N_TRACK)
        if self._R is None:
            self._R = torch.eye(3, device=self.device).expand(self.E, 3, 3).contiguous()
            self._t = torch.zeros(self.E, 3, device=self.device)
        warm = self._pts0 @ self._R.transpose(1, 2) + self._t.unsqueeze(1)
        dR, dt = icp(warm, cur, valid, iters=ICP_ITERS, trim=ICP_TRIM, reg=ICP_REG)
        self._R = torch.where(valid[:, None, None], dR @ self._R, self._R)
        self._t = torch.where(valid[:, None], (dR @ self._t.unsqueeze(-1)).squeeze(-1) + dt, self._t)
        return self._pts0 @ self._R.transpose(1, 2) + self._t.unsqueeze(1)

    def _observed_displacement(self, tracked):
        """Centroid displacement of the movable part since reset."""
        d = tracked.mean(1) - self._cent0
        if self._given0 is None or self._given is None:
            return d
        pris = self.pf.is_prismatic()
        d = d.clone()
        for e in range(self.E):
            if not bool(pris[e]):
                continue
            m0 = (self._given0[e] == 1) | (self._given0[e] == 2)
            m1 = (self._given[e] == 1) | (self._given[e] == 2)
            if m0.any() and m1.any():
                dd = self._xyz[e][m1].mean(0) - self._xyz0[e][m0].mean(0)
                if dd.norm() > 1e-4:
                    d[e] = dd
        return d

    def _propose_prismatic(self, d, todo):
        """Add prismatic candidates along the observed displacement direction and a small fan
        around it. Safe for doors: a line cannot follow an arc, so these lose on a door."""
        fan = MOTION_FAN_DEG * math.pi / 180.0
        dhat = d / (d.norm(dim=-1, keepdim=True) + 1e-9)
        for e in range(self.E):
            if not bool(todo[e]):
                continue
            u0 = dhat[e]
            a = torch.tensor([0., 0., 1.], device=self.device)
            if abs(float(u0[2])) > 0.9:
                a = torch.tensor([1., 0., 0.], device=self.device)
            e1 = torch.cross(u0, a, dim=-1)
            e1 = e1 / (e1.norm() + 1e-9)
            e2 = torch.cross(u0, e1, dim=-1)
            free = self.pf.free_slots(e)
            if free.numel() == 0:
                continue
            w = self.pf.median_weight(e, 1.0 / MOTION_SEEDS)
            for j in range(min(MOTION_SEEDS, int(free.numel()))):
                if j == 0:
                    u = u0
                else:
                    phi = 2 * math.pi * (j - 1) / max(MOTION_SEEDS - 1, 1)
                    u = u0 * math.cos(fan) + (math.cos(phi) * e1 + math.sin(phi) * e2) * math.sin(fan)
                    u = u / (u.norm() + 1e-9)
                self.pf.add(e, int(free[j]), u, self._cent0[e], PRISMATIC, w)
            self.pf.normalize(e)

    def _propose_screw(self):
        """Add a revolute candidate from the screw decomposition of the tracked motion."""
        u, c, ok = screw_axis(self._R, self._t, SCREW_MIN_DEG)
        pris = self.pf.is_prismatic()
        for e in range(self.E):
            if not bool(ok[e]) or bool(pris[e]):
                continue
            s = int(self._screw_slot[e])
            if s < 0:
                free = self.pf.free_slots(e)
                if free.numel() == 0:
                    continue
                s = int(free[0])
                self._screw_slot[e] = s
            if self.pf.add(e, s, u[e] / (u[e].norm() + 1e-9), c[e], REVOLUTE, self.pf.median_weight(e, 0.05)):
                self.pf.normalize(e)

    def _reweight(self, tracked):
        """Propose candidates, reweight by the likelihood and resample."""
        d = self._observed_displacement(tracked)
        dist = d.norm(dim=-1)

        if self._seeded is None:
            self._seeded = torch.zeros(self.E, dtype=torch.bool, device=self.device)
        todo = (dist > MIN_DISP) & (~self._seeded)
        if bool(todo.any()):
            self._propose_prismatic(d, todo)
            self._seeded |= todo
            prev = self._seed_disp if self._seed_disp is not None else torch.zeros(self.E, device=self.device)
            self._seed_disp = torch.where(todo, dist, prev)
        if self._seed_disp is not None:
            again = self._seeded & (dist > self._seed_disp * RESEED_GROWTH)
            if bool(again.any()):
                self._propose_prismatic(d, again)
                self._seed_disp = torch.where(again, dist, self._seed_disp)
        self._propose_screw()

        pf = self.pf
        U, P, T = pf.U, pf.P, pf.T
        dd = d.unsqueeze(1)
        dn2 = (dd * dd).sum(-1)
        dU = (dd * U).sum(-1)
        # Residual of the centroid displacement: off the circle (revolute), off the line (prismatic), any (rigid)
        a = self._cent0.unsqueeze(1) - P
        aperp = a - (a * U).sum(-1, keepdim=True) * U
        r = aperp.norm(dim=-1)
        e1 = aperp / (r.unsqueeze(-1) + 1e-9)
        e2 = torch.cross(U, e1, dim=-1)
        al, be = (dd * e1).sum(-1), (dd * e2).sum(-1)
        res_rev = (torch.sqrt((al + r) ** 2 + be ** 2) - r) ** 2 + dU ** 2
        res_pris = dn2 - dU ** 2
        res = torch.where(T == REVOLUTE, res_rev, torch.where(T == PRISMATIC, res_pris, dn2))
        # Revolute: the cluster's covariance must rotate like R(u, theta) Cov0 R^T
        c_now = tracked.mean(1)
        Xc = tracked - c_now.unsqueeze(1)
        cov_now = Xc.transpose(1, 2) @ Xc / tracked.shape[1]
        th = torch.atan2(be, al + r)
        sk = torch.zeros(*U.shape[:2], 3, 3, device=U.device)
        sk[..., 0, 1] = -U[..., 2]; sk[..., 0, 2] = U[..., 1]; sk[..., 1, 2] = -U[..., 0]
        sk = sk - sk.transpose(-1, -2)
        R = torch.eye(3, device=U.device) + torch.sin(th)[..., None, None] * sk \
            + (1 - torch.cos(th))[..., None, None] * (sk @ sk)
        cov_pred = R @ self._cov0.unsqueeze(1) @ R.transpose(-1, -2)
        res_cov = ((cov_now.unsqueeze(1) - cov_pred) ** 2).sum((-1, -2))
        scale = (self._cov0 ** 2).sum((-1, -2)).clamp_min(1e-9)
        res = res + COV_WEIGHT * (SIGMA ** 2) * res_cov / scale.unsqueeze(1) * (T == REVOLUTE).float()

        log_lik = (-0.5 * res / (SIGMA ** 2)).masked_fill(~pf.valid, float("-inf"))
        log_w = (torch.log(pf.W.clamp(min=1e-30)) + log_lik).masked_fill(~pf.valid, float("-inf"))
        moved = dist > MIN_DISP
        pf.W = torch.where(moved.unsqueeze(-1), torch.softmax(log_w, dim=1), pf.W)
        self._resample(moved)

    def _resample(self, moved):
        """ESS-gated resampling with axis and pivot jitter."""
        pf = self.pf
        E, K = pf.W.shape
        valid = pf.valid
        nval = valid.sum(1)
        W = pf.W * valid.float()
        W = W / W.sum(1, keepdim=True).clamp_min(1e-12)
        ess = 1.0 / (W ** 2).sum(1).clamp_min(1e-12)
        do = moved & (nval > 1) & (ess < ESS_FRAC * nval.float())
        if not bool(do.any()):
            return
        protected = torch.zeros_like(valid)
        ei = (self._screw_slot >= 0).nonzero(as_tuple=True)[0]
        if ei.numel():
            protected[ei, self._screw_slot[ei]] = True
        pool = valid & ~protected
        do = do & (pool.sum(1) > 1)
        if not bool(do.any()):
            return
        Ws = torch.where(valid, W, torch.zeros_like(W))
        Ws = torch.where(Ws.sum(1, keepdim=True) > 0, Ws, torch.ones_like(Ws))
        idx = torch.multinomial(Ws, K, replacement=True)
        g3 = idx.unsqueeze(-1).expand(-1, -1, 3)
        Un, Pn, Tn = torch.gather(pf.U, 1, g3), torch.gather(pf.P, 1, g3), torch.gather(pf.T, 1, idx)
        if self._jitter is None:
            self._jitter = torch.ones(E, device=self.device)
        sc = self._jitter.view(-1, 1, 1)
        Un = Un + JITTER_AXIS * sc * torch.randn_like(Un)
        Un = Un / (Un.norm(dim=-1, keepdim=True) + 1e-9)
        noise = JITTER_PIVOT * sc * torch.randn_like(Pn)
        noise = noise - (noise * Un).sum(-1, keepdim=True) * Un       # Moving p along u changes nothing
        Pn = Pn + noise
        write = do.view(-1, 1) & pool
        pf.U = torch.where(write.unsqueeze(-1), Un, pf.U)
        pf.P = torch.where(write.unsqueeze(-1), Pn, pf.P)
        pf.T = torch.where(write, Tn, pf.T)
        w_pool = (pf.W * write.float()).sum(1, keepdim=True)
        n_pool = write.float().sum(1, keepdim=True).clamp_min(1.0)
        Wn = torch.where(write, (w_pool / n_pool).expand_as(pf.W), pf.W) * valid.float()
        Wn = Wn / Wn.sum(1, keepdim=True).clamp_min(1e-12)
        pf.W = torch.where(do.view(-1, 1), Wn, pf.W)
        self._jitter = torch.where(do, (self._jitter * JITTER_DECAY).clamp_min(JITTER_FLOOR), self._jitter)

    # ------------------------------------------------------------------ Flow observation
    def _reference_axis(self):
        """Current axis estimate, used to give the flow a consistent sign."""
        pris = self.pf.is_prismatic()
        u = self.pf.mean_axis_all()
        if bool(pris.any()):
            u = torch.where(pris.view(-1, 1), self.pf.mean_axis_of_type(pris), u)
            u = torch.where(pris.view(-1, 1), self.pf.best_axis(u), u)
        return u

    def flow(self, xyz, movable):
        """Articulation flow [E, N, 3] on the movable points: per-type fields weighted by the type belief."""
        xyz = xyz.to(self.device).float()
        movable = movable.to(self.device)
        pf = self.pf
        E, N, _ = xyz.shape
        u_ref = self._reference_axis()
        W_rev = pf.W * ((pf.T == REVOLUTE) & pf.valid).float()
        W_pris = pf.W * ((pf.T == PRISMATIC) & pf.valid).float()
        m_rev, m_pris = W_rev.sum(1, keepdim=True), W_pris.sum(1, keepdim=True)
        total = (m_rev + m_pris).clamp(min=1e-9)
        U_rev = pf.U * torch.sign((pf.U * self._group_ref(W_rev, u_ref).unsqueeze(1)).sum(-1, keepdim=True) + 1e-9)
        U_pris = pf.U * torch.sign((pf.U * self._group_ref(W_pris, u_ref).unsqueeze(1)).sum(-1, keepdim=True) + 1e-9)
        # Revolute: E_k[u_k x (x - p_k)] = E[u] x x - E[u x p] (bilinear, no [E, K, N, 3] tensor)
        w = (W_rev / m_rev.clamp(min=1e-9)).unsqueeze(-1)
        u_bar = (w * U_rev).sum(dim=1)
        c_bar = (w * torch.cross(U_rev, pf.P, dim=-1)).sum(dim=1)
        f_rev = torch.cross(u_bar.unsqueeze(1).expand(-1, N, -1), xyz, dim=-1) - c_bar.unsqueeze(1)
        f_rev = torch.where((m_rev > 1e-9).view(-1, 1, 1), f_rev, torch.zeros_like(f_rev))
        # Prismatic: constant field E_k[u_k]
        w = (W_pris / m_pris.clamp(min=1e-9)).unsqueeze(-1)
        f_pris = (w * U_pris).sum(dim=1).unsqueeze(1).expand(-1, N, -1)
        f_pris = torch.where((m_pris > 1e-9).view(-1, 1, 1), f_pris, torch.zeros_like(f_pris))

        mask = movable.unsqueeze(-1).float()
        out = 0
        for f, mass in ((f_rev, m_rev / total), (f_pris, m_pris / total)):
            g = f / _mean_norm(f, mask)
            m = g.norm(dim=-1, keepdim=True)
            g = torch.nan_to_num(g / (m + 1e-9)) * (1.0 + FLOW_BETA * (m - 1.0)).clamp(min=0.0)
            n = g.norm(dim=-1, keepdim=True)
            g = g * (n.clamp(max=FLOW_MAX_NORM) / (n + 1e-9))
            out = out + torch.nan_to_num(g * mask * mass.view(-1, 1, 1))
        return out

    def _group_ref(self, W_group, u_ref):
        """Sign reference for one type group (axes are undirected)."""
        ref = self.pf.U[torch.arange(self.E, device=self.device), W_group.argmax(dim=1)]
        c = (ref * u_ref).sum(-1, keepdim=True)
        s = torch.where(c.abs() > 0.5, c, torch.ones_like(c))
        return ref * torch.sign(s + 1e-9)

    # ------------------------------------------------------------------ Information gain
    def entropy(self):
        """Belief entropy in nats: joint-type term plus a weighted KDE over the particles."""
        pf = self.pf
        vl = pf.valid
        Wa = pf.W * vl.float()
        Wa = Wa / Wa.sum(1, keepdim=True).clamp(min=1e-30)
        Un = pf.U / (pf.U.norm(dim=-1, keepdim=True) + 1e-9)
        H = torch.zeros(self.E, device=self.device)
        # Scott's rule with the intrinsic dimension (prismatic 2, revolute 4)
        for t, dof in ((PRISMATIC, 2), (REVOLUTE, 4)):
            m = (pf.T == t) & vl
            if not bool(m.any()):
                continue
            th = Un if t == PRISMATIC else torch.cat([Un, pf.P], dim=-1)
            D = th.shape[-1]
            w = Wa * m.float()
            s = w.sum(1, keepdim=True)
            wn = w / s.clamp(min=1e-30)
            mu = (wn.unsqueeze(-1) * th).sum(1, keepdim=True)
            sd = (wn.unsqueeze(-1) * (th - mu) ** 2).sum(1, keepdim=True).clamp(min=1e-12).sqrt()
            n = m.float().sum(1).clamp(min=2.0).view(-1, 1, 1)
            h = ((n ** (-1.0 / (dof + 4))) * sd).clamp(min=1e-6)
            d2 = ((th.unsqueeze(2) - th.unsqueeze(1)) / h.unsqueeze(1)).pow(2).sum(-1)
            lnorm = (0.5 * D * math.log(2 * math.pi) + h.log().sum(-1)).unsqueeze(-1)
            lk = (-0.5 * d2 - lnorm).masked_fill(~m.unsqueeze(1), float("-inf"))
            lf = torch.logsumexp(lk + wn.clamp(min=1e-30).log().unsqueeze(1), dim=2)
            lf = torch.where(m, lf, torch.zeros_like(lf))
            H = H + s.squeeze(1) * (-(wn * torch.nan_to_num(lf, nan=0.0, posinf=0.0, neginf=0.0)).sum(1))
        Pt = torch.stack([(Wa * ((pf.T == t) & vl).float()).sum(1) for t in (0, 1, 2)], dim=1)
        Pt = (Pt / Pt.sum(1, keepdim=True).clamp(min=1e-30)).clamp(min=1e-30)
        return H - (Pt * Pt.log()).sum(1)

    def information_gain(self):
        """Entropy decrease since the previous call."""
        H = self.entropy().detach()
        gain = torch.zeros_like(H) if self._ent_prev is None else self._ent_prev - H
        self._ent_prev = H
        return gain


def _mean_norm(f, mask):
    """Per-env mean flow length over the movable points, broadcast to [E, N, 1]."""
    n = f.norm(dim=-1, keepdim=True)
    m = (n * mask).sum(dim=1, keepdim=True) / mask.sum(dim=1, keepdim=True).clamp(min=1.0)
    return torch.where(m > 1e-9, m, torch.ones_like(m)).expand_as(n)
