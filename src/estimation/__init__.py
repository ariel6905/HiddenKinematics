"""Articulation estimation: movable-part tracking, particle-filter belief and articulation flow."""
from .estimator import ArticulationEstimator
from .prior import PRIOR_MODES, load_prior
from .segmentation import MovableTracker


class ArticulationFlow:
    """Movable segmentation, articulation belief and flow channels of the policy (``reset``, then ``step``)."""

    def __init__(self, task, prior, device, prior_mode="densify"):
        self.task = task
        self.tracker = MovableTracker(device)
        self.estimator = ArticulationEstimator(task, prior, device, prior_mode)

    def reset(self, xyz, given_seg=None):
        movable = self.tracker.reset(self.task, xyz)
        self.estimator.reset(xyz, movable, given_seg)
        return self.estimator.flow(xyz, movable)

    def step(self, xyz, given_seg=None):
        movable = self.tracker.step(xyz)
        self.estimator.update(xyz, movable, given_seg)
        return self.estimator.flow(xyz, movable)


__all__ = ["ArticulationFlow", "ArticulationEstimator", "MovableTracker", "PRIOR_MODES", "load_prior"]
