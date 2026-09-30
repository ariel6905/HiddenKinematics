"""PartManip cabinet environment (import before torch, as IsaacGym requires)."""
import copy
import os

# CUDA and Vulkan (cameras) must enumerate GPUs in the same (PCI) order
os.environ.setdefault("CUDA_DEVICE_ORDER", "PCI_BUS_ID")

import yaml  # noqa: E402
from isaacgym import gymapi  # noqa: E402

from .utils.load_env import SPLITS, default_asset_root, list_assets, read_shard
from .utils.misc import PHYSX, SIM_DT

_CFG_PATH = os.path.join(os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__)))), "configs", "env.yaml")


def env_config(preset, asset_root=None, **overrides):
    """Return the merged configuration dict of ``preset`` ("teacher" or "student")."""
    with open(_CFG_PATH) as f:
        spec = yaml.safe_load(f)
    cfg = copy.deepcopy(spec["common"])
    cfg.update(copy.deepcopy(spec[preset]))
    cfg["asset_root"] = os.path.abspath(asset_root or default_asset_root())
    cfg.update(overrides)
    return cfg


def sim_params():
    """PhysX simulation parameters (GPU pipeline)."""
    p = gymapi.SimParams()
    p.dt = SIM_DT
    p.num_client_threads = 0
    for k, v in PHYSX.items():
        setattr(p.physx, k, v)
    p.physx.use_gpu = True
    p.physx.num_subscenes = 0
    p.use_gpu_pipeline = True
    return p


def make_env(preset, assets, device_id=0, asset_root=None, **overrides):
    """Build the environment; ``assets`` maps "train" / "val" to lists of asset directories."""
    from .franka_pose_cabinet_base import CabinetEnv
    assets = {s: list(assets.get(s, [])) for s in SPLITS}
    return CabinetEnv(env_config(preset, asset_root, **overrides), sim_params(), assets, device_id)


__all__ = ["make_env", "env_config", "list_assets", "read_shard", "default_asset_root"]
