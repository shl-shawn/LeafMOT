import os
from collections import OrderedDict
from typing import List, Optional, Tuple, Union

import torch
import torch.nn as nn
import torch.nn.functional as F


# ============================================================
# Minimal replacements for mmcv/mmengine helpers
# ============================================================

class DropPath(nn.Module):
    """DropPath / Stochastic Depth (timm-style, minimal local version)."""

    def __init__(self, drop_prob: float = 0.0):
        super().__init__()
        self.drop_prob = float(drop_prob)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        if self.drop_prob == 0.0 or not self.training:
            return x
        keep_prob = 1.0 - self.drop_prob
        shape = (x.shape[0],) + (1,) * (x.ndim - 1)
        random_tensor = keep_prob + torch.rand(shape, dtype=x.dtype, device=x.device)
        random_tensor.floor_()
        return x.div(keep_prob) * random_tensor


def build_norm_layer(num_features: int) -> nn.Module:
    """
    Official DFormer uses build_norm_layer(norm_cfg, channels)[1].
    In practice for our use, BatchNorm2d is enough and avoids mmcv.
    """
    return nn.BatchNorm2d(num_features)


# ============================================================
# Official DFormer encoder blocks (adapted with minimal changes)
# Source logic: VCIP-RGBD/DFormer official encoder
# ============================================================

class LayerNorm(nn.Module):
    """
    LayerNorm that supports channels_last and channels_first.
    Kept identical in spirit to the official implementation.
    """

    def __init__(self, normalized_shape: int, eps: float = 1e-6, data_format: str = "channels_last"):
        super().__init__()
        self.weight = nn.Parameter(torch.ones(normalized_shape))
        self.bias = nn.Parameter(torch.zeros(normalized_shape))
        self.eps = eps
        self.data_format = data_format
        if self.data_format not in ["channels_last", "channels_first"]:
            raise NotImplementedError(f"Unsupported data_format: {self.data_format}")
        self.normalized_shape = (normalized_shape,)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        if self.data_format == "channels_last":
            return F.layer_norm(x, self.normalized_shape, self.weight, self.bias, self.eps)

        # channels_first
        u = x.mean(1, keepdim=True)
        s = (x - u).pow(2).mean(1, keepdim=True)
        x = (x - u) / torch.sqrt(s + self.eps)
        x = self.weight[:, None, None] * x + self.bias[:, None, None]
        return x


class DFormerMLP(nn.Module):
    def __init__(self, dim: int, mlp_ratio: int = 4):
        super().__init__()
        self.norm = LayerNorm(dim, eps=1e-6, data_format="channels_last")
        self.fc1 = nn.Linear(dim, dim * mlp_ratio)
        self.pos = nn.Conv2d(dim * mlp_ratio, dim * mlp_ratio, kernel_size=3, padding=1, groups=dim * mlp_ratio)
        self.fc2 = nn.Linear(dim * mlp_ratio, dim)
        self.act = nn.GELU()

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        # x: [B, H, W, C]
        x = self.norm(x)
        x = self.fc1(x)
        x = x.permute(0, 3, 1, 2)              # [B, C', H, W]
        x = self.pos(x) + x
        x = x.permute(0, 2, 3, 1)              # [B, H, W, C']
        x = self.act(x)
        x = self.fc2(x)
        return x


class DFormerAttention(nn.Module):
    """
    Official DFormer attention block, adapted from the released code.
    """

    def __init__(self, dim: int, num_head: int = 8, window: int = 7, drop_depth: bool = False):
        super().__init__()
        self.num_head = num_head
        self.window = window
        self.drop_depth = drop_depth

        self.q = nn.Linear(dim, dim)
        self.q_cut = nn.Linear(dim, dim // 2)

        self.a = nn.Linear(dim, dim)
        self.l = nn.Linear(dim, dim)

        self.conv = nn.Conv2d(dim, dim, kernel_size=7, padding=3, groups=dim)

        self.e_conv = nn.Conv2d(dim // 2, dim // 2, kernel_size=7, padding=3, groups=dim // 2)
        self.e_fore = nn.Linear(dim // 2, dim // 2)
        self.e_back = nn.Linear(dim // 2, dim // 2)

        # default case (window == 0)
        self.proj = nn.Linear(dim // 2 * 3, dim)  # adjusted below if window != 0
        if not drop_depth:
            self.proj_e = nn.Linear(dim // 2 * 3, dim // 2)

        if window != 0:
            # official code concatenates [x, attn, cutted_x] => dim + dim/2 + dim/2 = 2*dim
            self.short_cut_linear = nn.Linear(dim + dim // 2, dim // 2)
            self.kv = nn.Linear(dim, dim)
            self.pool = nn.AdaptiveAvgPool2d(output_size=(window, window))
            self.proj = nn.Linear(dim * 2, dim)
            if not drop_depth:
                self.proj_e = nn.Linear(dim * 2, dim // 2)

        self.act = nn.GELU()
        self.norm = LayerNorm(dim, eps=1e-6, data_format="channels_last")
        self.norm_e = LayerNorm(dim // 2, eps=1e-6, data_format="channels_last")

    def forward(self, x: torch.Tensor, x_e: torch.Tensor) -> Tuple[torch.Tensor, Optional[torch.Tensor]]:
        # x:   [B, H, W, C]
        # x_e: [B, H, W, C/2]
        B, H, W, C = x.shape

        x = self.norm(x)
        x_e = self.norm_e(x_e)

        if self.window != 0:
            short_cut = torch.cat([x, x_e], dim=3)          # [B, H, W, C + C/2]
            short_cut = short_cut.permute(0, 3, 1, 2)       # [B, C + C/2, H, W]

        q = self.q(x)                                       # [B, H, W, C]
        cutted_x = self.q_cut(x)                            # [B, H, W, C/2]

        x_local = self.l(x).permute(0, 3, 1, 2)             # [B, C, H, W]
        x_local = self.act(x_local)

        a = self.conv(x_local)                              # [B, C, H, W]
        a = a.permute(0, 2, 3, 1)                           # [B, H, W, C]
        a = self.a(a)                                       # [B, H, W, C]

        if self.window != 0:
            b = x_local.permute(0, 2, 3, 1)                 # [B, H, W, C]
            kv = self.kv(b)
            kv = kv.reshape(B, H * W, 2, self.num_head, C // self.num_head // 2).permute(2, 0, 3, 1, 4)
            k, v = kv.unbind(0)                             # each: [B, heads, HW, C/(2h)]

            short_cut = self.pool(short_cut).permute(0, 2, 3, 1)  # [B, window, window, C + C/2]
            short_cut = self.short_cut_linear(short_cut)           # [B, window, window, C/2]
            short_cut = short_cut.reshape(B, -1, self.num_head, C // self.num_head // 2).permute(0, 2, 1, 3)

            m = short_cut                                   # [B, heads, window*window, C/(2h)]
            attn = (m * (C // self.num_head // 2) ** -0.5) @ k.transpose(-2, -1)
            attn = attn.softmax(dim=-1)
            attn = (
                (attn @ v)
                .reshape(B, self.num_head, self.window, self.window, C // self.num_head // 2)
                .permute(0, 1, 4, 2, 3)
                .reshape(B, C // 2, self.window, self.window)
            )
            attn = F.interpolate(attn, size=(H, W), mode="bilinear", align_corners=False)
            attn = attn.permute(0, 2, 3, 1)                 # [B, H, W, C/2]

        x_e_out = self.e_back(
            self.e_conv(self.e_fore(x_e).permute(0, 3, 1, 2)).permute(0, 2, 3, 1)
        )                                                   # [B, H, W, C/2]

        cutted_x = cutted_x * x_e_out                       # [B, H, W, C/2]
        x_main = q * a                                      # [B, H, W, C]

        if self.window != 0:
            x_cat = torch.cat([x_main, attn, cutted_x], dim=3)   # [B, H, W, 2C]
        else:
            x_cat = torch.cat([x_main, cutted_x], dim=3)         # [B, H, W, C + C/2] but official proj dims handle no-window stage

        if not self.drop_depth:
            x_e_out = self.proj_e(x_cat)
        else:
            x_e_out = x_e

        x_out = self.proj(x_cat)
        return x_out, x_e_out


class DFormerBlock(nn.Module):
    def __init__(
        self,
        index: int,
        dim: int,
        num_head: int,
        mlp_ratio: float = 4.0,
        block_index: int = 0,
        last_block_index: int = 50,
        window: int = 7,
        drop_prob: float = 0.0,
        drop_depth: bool = False,
    ):
        super().__init__()
        self.index = index
        layer_scale_init_value = 1e-6

        if block_index > last_block_index:
            window = 0

        self.attn = DFormerAttention(dim=dim, num_head=num_head, window=window, drop_depth=drop_depth)
        self.mlp = DFormerMLP(dim=dim, mlp_ratio=int(mlp_ratio))

        self.dropout_layer = DropPath(drop_prob) if drop_prob > 0 else nn.Identity()

        self.layer_scale_1 = nn.Parameter(layer_scale_init_value * torch.ones(dim), requires_grad=True)
        self.layer_scale_2 = nn.Parameter(layer_scale_init_value * torch.ones(dim), requires_grad=True)

        self.drop_depth = drop_depth
        if not drop_depth:
            self.layer_scale_1_e = nn.Parameter(layer_scale_init_value * torch.ones(dim // 2), requires_grad=True)
            self.layer_scale_2_e = nn.Parameter(layer_scale_init_value * torch.ones(dim // 2), requires_grad=True)
            self.mlp_e2 = DFormerMLP(dim=dim // 2, mlp_ratio=int(mlp_ratio))

    def forward(self, x: torch.Tensor, x_e: torch.Tensor) -> Tuple[torch.Tensor, Optional[torch.Tensor]]:
        # x, x_e: channels_last
        res_x, res_e = x, x_e
        x, x_e = self.attn(x, x_e)

        x = res_x + self.dropout_layer(self.layer_scale_1.view(1, 1, 1, -1) * x)
        x = x + self.dropout_layer(self.layer_scale_2.view(1, 1, 1, -1) * self.mlp(x))

        if not self.drop_depth:
            x_e = res_e + self.dropout_layer(self.layer_scale_1_e.view(1, 1, 1, -1) * x_e)
            x_e = x_e + self.dropout_layer(self.layer_scale_2_e.view(1, 1, 1, -1) * self.mlp_e2(x_e))

        return x, x_e


class OfficialDFormerEncoder(nn.Module):
    """
    Official DFormer encoder architecture, minimally adapted to remove mmcv/mmengine.
    """

    def __init__(
        self,
        depths: Tuple[int, int, int, int],
        dims: Tuple[int, int, int, int],
        windows: Tuple[int, int, int, int],
        mlp_ratios: Tuple[int, int, int, int],
        num_heads: Tuple[int, int, int, int],
        last_block: Tuple[int, int, int, int],
        drop_path_rate: float = 0.1,
    ):
        super().__init__()

        self.depths = depths
        self.dims = dims

        # RGB downsampling path
        self.downsample_layers = nn.ModuleList()
        stem = nn.Sequential(
            nn.Conv2d(3, dims[0] // 2, kernel_size=3, stride=2, padding=1),
            nn.BatchNorm2d(dims[0] // 2),
            nn.GELU(),
            nn.Conv2d(dims[0] // 2, dims[0], kernel_size=3, stride=2, padding=1),
            nn.BatchNorm2d(dims[0]),
        )
        self.downsample_layers.append(stem)

        # Depth downsampling path
        self.downsample_layers_e = nn.ModuleList()
        stem_e = nn.Sequential(
            nn.Conv2d(1, dims[0] // 4, kernel_size=3, stride=2, padding=1),
            nn.BatchNorm2d(dims[0] // 4),
            nn.GELU(),
            nn.Conv2d(dims[0] // 4, dims[0] // 2, kernel_size=3, stride=2, padding=1),
            nn.BatchNorm2d(dims[0] // 2),
        )
        self.downsample_layers_e.append(stem_e)

        for i in range(len(dims) - 1):
            self.downsample_layers.append(
                nn.Sequential(
                    build_norm_layer(dims[i]),
                    nn.Conv2d(dims[i], dims[i + 1], kernel_size=3, stride=2, padding=1),
                )
            )
            self.downsample_layers_e.append(
                nn.Sequential(
                    build_norm_layer(dims[i] // 2),
                    nn.Conv2d(dims[i] // 2, dims[i + 1] // 2, kernel_size=3, stride=2, padding=1),
                )
            )

        # stages
        self.stages = nn.ModuleList()
        dp_rates = torch.linspace(0, drop_path_rate, sum(depths)).tolist()
        cur = 0
        for i in range(len(dims)):
            stage = nn.ModuleList(
                [
                    DFormerBlock(
                        index=cur + j,
                        dim=dims[i],
                        num_head=num_heads[i],
                        mlp_ratio=mlp_ratios[i],
                        block_index=depths[i] - j,
                        last_block_index=last_block[i],
                        window=windows[i],
                        drop_prob=dp_rates[cur + j],
                        drop_depth=((i == 3) and (j == depths[i] - 1)),
                    )
                    for j in range(depths[i])
                ]
            )
            self.stages.append(stage)
            cur += depths[i]

    def load_pretrained(self, ckpt_path: str, strict: bool = False) -> None:
        """
        Compatible with the official DFormer checkpoint loading style.
        Official code strips 'backbone.' and optionally 'module.' prefixes.
        """
        state = torch.load(ckpt_path, map_location="cpu", weights_only=False)
        if isinstance(state, dict):
            if "state_dict_ema" in state:
                state = state["state_dict_ema"]
            elif "state_dict" in state:
                state = state["state_dict"]

        new_state = OrderedDict()
        for k, v in state.items():
            if k.startswith("backbone."):
                k = k[9:]
            if k.startswith("module."):
                k = k[7:]
            new_state[k] = v

        missing, unexpected = self.load_state_dict(new_state, strict=strict)
        print(f"[DFormer] Loaded checkpoint: {ckpt_path}")
        print(f"[DFormer] Missing keys: {len(missing)}")
        print(f"[DFormer] Unexpected keys: {len(unexpected)}")

    def forward(self, rgb: torch.Tensor, depth: Optional[torch.Tensor]) -> List[torch.Tensor]:
        """
        Args:
            rgb:   [B, 3, H, W]
            depth: [B, 1, H, W] or [B, C, H, W] (only first channel will be used)

        Returns:
            outs: [c1, c2, c3, c4]
        """
        if depth is None:
            # fallback: use RGB luminance-like proxy (not recommended)
            depth = rgb[:, :1, :, :]

        if depth.ndim == 3:
            depth = depth.unsqueeze(1)
        if rgb.ndim == 3:
            rgb = rgb.unsqueeze(0)

        depth = depth[:, :1, :, :]

        outs = []
        x = rgb
        x_e = depth

        for i in range(4):
            x = self.downsample_layers[i](x)
            x_e = self.downsample_layers_e[i](x_e)

            x = x.permute(0, 2, 3, 1)     # [B,H,W,C]
            x_e = x_e.permute(0, 2, 3, 1) # [B,H,W,C/2]

            for blk in self.stages[i]:
                x, x_e = blk(x, x_e)

            x = x.permute(0, 3, 1, 2).contiguous()
            x_e = x_e.permute(0, 3, 1, 2).contiguous()

            outs.append(x)

        return outs


def build_dformer_tiny_encoder(drop_path_rate: float = 0.1) -> OfficialDFormerEncoder:
    """
    Official DFormer-Tiny stage config:
    dims=[32, 64, 128, 256]
    depths=[3, 3, 5, 2]
    num_heads=[1, 2, 4, 8]
    windows=[0, 7, 7, 7]
    mlp_ratios=[8, 8, 4, 4]
    """
    return OfficialDFormerEncoder(
        dims=(32, 64, 128, 256),
        depths=(3, 3, 5, 2),
        num_heads=(1, 2, 4, 8),
        windows=(0, 7, 7, 7),
        mlp_ratios=(8, 8, 4, 4),
        last_block=(50, 50, 50, 50),
        drop_path_rate=drop_path_rate,
    )

def build_dformer_small_encoder(drop_path_rate: float = 0.1) -> OfficialDFormerEncoder:
    """
    Official DFormer-Tiny stage config:
    dims=[32, 64, 128, 256]
    depths=[3, 3, 5, 2]
    num_heads=[1, 2, 4, 8]
    windows=[0, 7, 7, 7]
    mlp_ratios=[8, 8, 4, 4]
    """
    return OfficialDFormerEncoder(
        dims=[64, 128, 256, 512],
        depths=[2, 2, 4, 2],
        num_heads=(1, 2, 4, 8),
        windows=(0, 7, 7, 7),
        mlp_ratios=(8, 8, 4, 4),
        last_block=(50, 50, 50, 50),
        drop_path_rate=drop_path_rate,
    )

# ============================================================
# Lightweight tracking neck (c2, c3, c4 -> stride-8 fused map)
# ============================================================

class ConvBNAct(nn.Module):
    def __init__(self, in_ch: int, out_ch: int, k: int = 1, s: int = 1, p: int = 0, groups: int = 1):
        super().__init__()
        self.block = nn.Sequential(
            nn.Conv2d(in_ch, out_ch, kernel_size=k, stride=s, padding=p, groups=groups, bias=False),
            nn.BatchNorm2d(out_ch),
            nn.SiLU(inplace=True),
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.block(x)


class TrackingFusionNeck(nn.Module):
    """
    DFormer-style lightweight multi-scale fusion,
    adapted for tracking:
      - use c2, c3, c4
      - fuse to c2 resolution (stride 8)
      - output one fused feature map for ROIAlign
    """

    def __init__(self, in_channels=(64, 128, 256), out_channels: int = 256):
        super().__init__()
        c2_ch, c3_ch, c4_ch = in_channels

        self.proj2 = ConvBNAct(c2_ch, out_channels, k=1, s=1, p=0)
        self.proj3 = ConvBNAct(c3_ch, out_channels, k=1, s=1, p=0)
        self.proj4 = ConvBNAct(c4_ch, out_channels, k=1, s=1, p=0)

        self.fuse = nn.Sequential(
            ConvBNAct(out_channels * 3, out_channels, k=1, s=1, p=0),
            ConvBNAct(out_channels, out_channels, k=3, s=1, p=1, groups=out_channels),  # depthwise
            ConvBNAct(out_channels, out_channels, k=1, s=1, p=0),
        )

    def forward(self, c2: torch.Tensor, c3: torch.Tensor, c4: torch.Tensor) -> torch.Tensor:
        """
        Inputs:
            c2: [B,  64, H/8,  W/8 ]
            c3: [B, 128, H/16, W/16]
            c4: [B, 256, H/32, W/32]

        Output:
            F:  [B, 256, H/8,  W/8 ]
        """
        u2 = self.proj2(c2)
        u3 = self.proj3(c3)
        u4 = self.proj4(c4)

        target_size = u2.shape[-2:]
        u3 = F.interpolate(u3, size=target_size, mode="bilinear", align_corners=False)
        u4 = F.interpolate(u4, size=target_size, mode="bilinear", align_corners=False)

        x = torch.cat([u2, u3, u4], dim=1)
        x = self.fuse(x)
        return x


# ============================================================
# Final Module 2 wrapper for LeafMOT
# ============================================================

class RGBDEncoder(nn.Module):
    """
    LeafMOT Module 2:
      Official DFormer-Tiny encoder + lightweight tracking neck

    Input:
        rgb   [B, 3, H, W]
        depth [B, 1, H, W]

    Output:
        fused [B, 256, H/8, W/8]
    """

    def __init__(
        self,
        out_channels: int = 256,
        drop_path_rate: float = 0.1,
        pretrained_path: Optional[str] = None,
        return_pyramid: bool = False,
    ):
        super().__init__()
        self.backbone = build_dformer_tiny_encoder(drop_path_rate=drop_path_rate)
        self.neck = TrackingFusionNeck(in_channels=(64, 128, 256), out_channels=out_channels)
        self.return_pyramid = return_pyramid

        if pretrained_path is not None and len(pretrained_path) > 0:
            self.backbone.load_pretrained(pretrained_path, strict=False)

    def forward(
        self,
        rgb: torch.Tensor,
        depth: torch.Tensor,
        return_pyramid: Optional[bool] = None,
    ) -> Union[torch.Tensor, Tuple[torch.Tensor, List[torch.Tensor]]]:
        feats = self.backbone(rgb, depth)  # [c1, c2, c3, c4]
        c1, c2, c3, c4 = feats

        fused = self.neck(c2, c3, c4)

        if return_pyramid is None:
            return_pyramid = self.return_pyramid

        if return_pyramid:
            return fused, feats
        return fused


# ============================================================
# Quick shape test
# ============================================================

if __name__ == "__main__":
    model = RGBDEncoder(out_channels=256, 
                        pretrained_path="/mnt/nfsshare/home/liu21/LeafMOT/LeafMOT/pretrained/DFormer_Tiny.pth.tar", 
                        return_pyramid=True)
    model.eval()

    rgb = torch.randn(1, 3, 1200, 1200)
    depth = torch.randn(1, 1, 1200, 1200)

    with torch.no_grad():
        fused, feats = model(rgb, depth, return_pyramid=True)

    print("c1:", feats[0].shape)  # expected ~ [1,  32, 300, 300]
    print("c2:", feats[1].shape)  # expected ~ [1,  64, 150, 150]
    print("c3:", feats[2].shape)  # expected ~ [1, 128,  75,  75]
    print("c4:", feats[3].shape)  # expected ~ [1, 256,  38,  38]
    print("fused:", fused.shape)  # expected ~ [1, 256, 150, 150]