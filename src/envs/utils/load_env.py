"""Asset lists of the PartManip dataset (``<asset_root>/<category>/<split>/<asset_name>/``)."""
import os
import re
import xml.etree.ElementTree as ET
from pathlib import Path

import yaml

SPLITS = ("train", "val")
_SPLIT_DIRS = {"train": ("train",), "val": ("val", "valIntra")}   # val: PartManip's valIntra

# Assets with broken simulation, excluded from every split (as in PartManip)
_BLACKLIST = {"door": ("StorageFurniture-46037-link_0-handle_0-joint_0-handlejoint_0",
                       "StorageFurniture-41083-link_1-handle_5-joint_1-handlejoint_5"),
              "drawer": ("StorageFurniture-48855-link_0-handle_0-joint_0-handlejoint_0",
                         "47207", "46537", "19855", "30666")}


def _target_joint_movable(asset_dir):
    """Check that the joint named in the asset name is movable in the URDF (true if none is named)."""
    m = re.search(r"(?<![a-z])(joint_\d+)", asset_dir.name)
    if not m:
        return True
    for urdf in ("mobility_new.urdf", "mobility.urdf"):
        path = asset_dir / urdf
        if path.exists():
            try:
                for j in ET.parse(str(path)).getroot().iter("joint"):
                    if j.get("name") == m.group(1):
                        return j.get("type") in ("revolute", "prismatic", "continuous")
            except ET.ParseError:
                return False
            return False
    return False


def usable(asset_dir, category):
    """Check that the asset has every file the environment needs and is not blacklisted."""
    if any(b in asset_dir.name for b in _BLACKLIST[category]):
        return False
    return ((asset_dir / "part_pregrasp_dof_state.npy").exists()
            and (asset_dir / "bbox_info.json").exists()
            and _target_joint_movable(asset_dir))


def list_assets(asset_root, category, split, names=None):
    """List the usable assets of one category and split, sorted by name."""
    dirs = [Path(asset_root) / category / d for d in _SPLIT_DIRS[split]]
    base = next((d for d in dirs if d.is_dir()), dirs[0])
    if names is not None:
        paths = [base / n for n in names]
        missing = [p for p in paths if not p.exists()]
        if missing:
            raise FileNotFoundError(f"{len(missing)} assets not found, e.g. {missing[0]}")
    else:
        paths = sorted(base.iterdir(), key=lambda p: p.name)
    return [p for p in paths if usable(p, category)]


def read_shard(path, category, shard):
    """Asset names of one training shard (``configs/shards.yaml``)."""
    with open(path) as f:
        return yaml.safe_load(f)[category][shard]


def default_asset_root():
    """``$ARTICULATE_ASSETS`` if set, otherwise ``<repo>/assets``."""
    root = os.path.join(os.path.dirname(os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))), "assets")
    return os.environ.get("ARTICULATE_ASSETS", root)
