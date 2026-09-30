"""Sparse 3D U-Net encoder (spconv) mapping a voxelized point cloud to one global feature."""
import functools

import spconv.pytorch as spconv
import torch
import torch.nn as nn
import torch.nn.functional as F
from spconv.pytorch.utils import PointToVoxel

VOXEL_SIZE = 0.009999999776482582        # 1 cm rounded to float32 (the grid of the released weights)


class ResBlock(spconv.SparseModule):
    """Residual block of two 3x3x3 submanifold convolutions."""

    def __init__(self, in_channels, out_channels, norm_fn, indice_key=None):
        super().__init__()
        if in_channels == out_channels:
            self.shortcut = nn.Identity()
        else:
            self.shortcut = spconv.SparseSequential(
                spconv.SubMConv3d(in_channels, out_channels, kernel_size=1, bias=False), norm_fn(out_channels))
        self.conv1 = spconv.SparseSequential(
            spconv.SubMConv3d(in_channels, out_channels, kernel_size=3, padding=1, bias=False, indice_key=indice_key),
            norm_fn(out_channels))
        self.conv2 = spconv.SparseSequential(
            spconv.SubMConv3d(out_channels, out_channels, kernel_size=3, padding=1, bias=False, indice_key=indice_key),
            norm_fn(out_channels))

    def forward(self, x):
        shortcut = self.shortcut(x)
        x = self.conv1(x)
        x = x.replace_feature(F.relu(x.features))
        x = self.conv2(x)
        return x.replace_feature(F.relu(x.features + shortcut.features))


class EncoderLevel(nn.Module):
    """One resolution level: residual blocks, then (except at the deepest level) a stride-2
    convolution into the next level. Parameter names match the released checkpoints."""

    def __init__(self, channels, block_repeat, norm_fn, level=1):
        super().__init__()
        self.channels = channels
        self.encoder_blocks = spconv.SparseSequential(
            *[ResBlock(channels[0], channels[0], norm_fn, indice_key=f"subm{level}") for _ in range(block_repeat)])
        if len(channels) > 1:
            self.downsample = spconv.SparseSequential(
                spconv.SparseConv3d(channels[0], channels[1], kernel_size=2, stride=2, bias=False,
                                    indice_key=f"spconv{level}"),
                norm_fn(channels[1]), nn.ReLU())
            self.ublock = EncoderLevel(channels[1:], block_repeat, norm_fn, level + 1)

    def forward(self, x):
        x = self.encoder_blocks(x)
        if len(self.channels) > 1:
            return self.ublock(self.downsample(x))
        return x


class SparseUNetEncoder(nn.Module):
    """Stem and encoder levels (channels 16 -> 64 -> 112); outputs [B, channels[-1]]."""

    def __init__(self, in_channels, channels=(16, 64, 112), block_repeat=2):
        super().__init__()
        norm_fn = functools.partial(nn.BatchNorm1d, eps=1e-4, momentum=0.1)
        self.stem = spconv.SparseSequential(
            spconv.SubMConv3d(in_channels, channels[0], kernel_size=3, padding=1, bias=False, indice_key="subm1"),
            norm_fn(channels[0]), nn.ReLU())
        self.ublock = EncoderLevel(list(channels), block_repeat, norm_fn)
        self.out_dim = channels[-1]

    def forward(self, pc):
        """Encode ``pc`` [B, N, 9] (xyz, rgb, flow); ``pc`` is modified in place."""
        B = pc.shape[0]
        pc[:, :, :3] = pc[:, :, :3] - pc[:, :, :3].mean(0, keepdim=True).mean(1, keepdim=True)
        pc[:, :, 0] = -pc[:, :, 0]
        pc[:, :, 6] = -pc[:, :, 6]
        feats, coords, batch_idx = voxelize(pc)
        spatial = (torch.max(coords, dim=0)[0] + 1).clamp(128, 100000000)
        coords = torch.cat((batch_idx.unsqueeze(1), coords), dim=1).int()
        x = spconv.SparseConvTensor(feats, coords, spatial_shape=spatial.tolist(), batch_size=B)
        x = self.ublock(self.stem(x))
        out = torch.zeros((B, x.features.shape[-1]), device=x.features.device, dtype=x.features.dtype)
        idx = x.indices[:, 0]
        for i in range(B):
            m = idx == i
            if m.any():
                out[i] = x.features[m].max(0)[0]
        return out


def voxelize(pc):
    """Mean-pool point features per occupied 1 cm voxel; returns (features, coords, cloud index)."""
    B, N, C = pc.shape
    xyz = pc[:, :, :3].reshape(-1, 3)
    lo = (xyz.min(0)[0] - 1e-4).tolist()
    hi = (xyz.max(0)[0] + 1e-4).tolist()
    grid = [max(1, int((hi[i] - lo[i]) / VOXEL_SIZE) + 1) for i in range(3)]
    gen = PointToVoxel(vsize_xyz=[VOXEL_SIZE] * 3, coors_range_xyz=lo + hi, num_point_features=C,
                       max_num_voxels=min(B * N, grid[0] * grid[1] * grid[2]), max_num_points_per_voxel=64,
                       device=pc.device)
    flat = pc.reshape(-1, C)
    feats, coords, batch_idx = [], [], []
    with torch.no_grad():
        for b in range(B):
            voxels, vcoords, num, _ = gen.generate_voxel_with_id(flat[b * N:(b + 1) * N].contiguous(), clear_voxels=True)
            if vcoords.shape[0] == 0:
                continue
            feats.append(voxels.sum(1) / num.clamp(min=1).unsqueeze(1).to(voxels.dtype))
            coords.append(vcoords[:, [2, 1, 0]].to(torch.int64))       # spconv (z, y, x) -> (x, y, z)
            batch_idx.append(torch.full((vcoords.shape[0],), b, dtype=torch.int64, device=pc.device))
    return torch.cat(feats, 0), torch.cat(coords, 0), torch.cat(batch_idx, 0)
