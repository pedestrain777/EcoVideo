#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Draw a clean entropy curve from existing WAN debug .pt files.
- No vertical lines (no keyframe markers)
- No frame/border
- No axes/ticks/labels
- Just a line

Supports:
- entropy_frame_step_XX.pt (per-step instantaneous curve, raw 1D tensor)
- final aggregated curve: WAN2.2 → entropy_keyframes.pt (dict["entropy"]),
  WAN2.1 → entropy_frame_final_{mode}.pt (raw tensor)

Example:
  # Plot step 4 instantaneous curve (last_frame):
  python tools/plot_entropy_curve_clean.py \
    --debug_dir outputs/entropy_scan_skate_24_32_v2/block_28 \
    --which step --step 4 \
    --out outputs/entropy_scan_skate_24_32_v2/block_28/entropy_curve_clean.png

  # Plot final EMA curve used for keyframe selection (from .pt, no keyframe markers):
  python tools/plot_entropy_curve_clean.py \
    --debug_dir outputs/entropy_scan_skate_24_32_v2/block_28 \
    --which final_ema \
    --out outputs/entropy_scan_skate_24_32_v2/block_28/entropy_curve_final_ema_clean.png
"""

import argparse
import os
from pathlib import Path
from typing import Any, Tuple, Optional

import numpy as np
import torch
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt


def _as_1d_tensor(obj: Any) -> torch.Tensor:
    """
    Make this robust to different torch.save formats:
      - Tensor
      - dict with common keys (e.g. entropy_keyframes.pt from WAN2.2)
      - list/tuple
    Returns: 1D float tensor on CPU.
    """
    if torch.is_tensor(obj):
        t = obj
    elif isinstance(obj, dict):
        for k in ["entropy", "ent", "values", "curve", "frame_entropy", "frame_attn_entropy", "y"]:
            if k in obj and torch.is_tensor(obj[k]):
                t = obj[k]
                break
        else:
            tens = [v for v in obj.values() if torch.is_tensor(v) and v.ndim == 1]
            if not tens:
                tens = [v for v in obj.values() if torch.is_tensor(v)]
            if not tens:
                raise TypeError(f"Unsupported dict format. Keys={list(obj.keys())}")
            t = tens[0]
    elif isinstance(obj, (list, tuple)):
        t = torch.tensor(obj)
    else:
        raise TypeError(f"Unsupported .pt content type: {type(obj)}")

    t = t.detach().float().cpu()
    if t.ndim == 0:
        t = t.view(1)
    elif t.ndim > 1:
        t = t.reshape(-1)
    return t


def _load_entropy_curve(debug_dir: str, which: str, step: int, mode: str = "ema") -> Tuple[torch.Tensor, Path]:
    d = Path(debug_dir)
    if which == "step":
        pt_path = d / f"entropy_frame_step_{step:02d}.pt"
        if not pt_path.is_file():
            raise FileNotFoundError(f"Cannot find: {pt_path}")
        obj = torch.load(pt_path, map_location="cpu")
        y = _as_1d_tensor(obj)
        return y, pt_path

    if which == "final_ema":
        # WAN2.2: entropy_keyframes.pt is a dict with "entropy" key
        pt_wan22 = d / "entropy_keyframes.pt"
        if pt_wan22.is_file():
            obj = torch.load(pt_wan22, map_location="cpu")
            y = _as_1d_tensor(obj)
            return y, pt_wan22
        # WAN2.1: entropy_frame_final_{mode}.pt is raw tensor
        pt_wan21 = d / f"entropy_frame_final_{mode}.pt"
        if pt_wan21.is_file():
            obj = torch.load(pt_wan21, map_location="cpu")
            y = _as_1d_tensor(obj)
            return y, pt_wan21
        raise FileNotFoundError(
            f"Cannot find final curve .pt in {d}. "
            f"Tried: {pt_wan22}, {pt_wan21}"
        )

    raise ValueError(f"Unknown --which {which}")


def plot_clean_line(
    y: torch.Tensor,
    out_path: str,
    dpi: int = 300,
    linewidth: float = 2.0,
    transparent: bool = True,
    figsize: Tuple[float, float] = (6.4, 4.8),
    y_pad_frac: float = 0.06,
    x_pad_frac: float = 0.02,
):
    """Only the line, no axes/frame; scaling and padding like original (图二) for similar look."""
    y_np = y.detach().float().cpu().numpy().reshape(-1)
    x = np.arange(len(y_np), dtype=np.float32)

    fig, ax = plt.subplots(figsize=figsize, dpi=dpi)
    ax.plot(x, y_np, linewidth=linewidth)

    ax.set_axis_off()
    for spine in ax.spines.values():
        spine.set_visible(False)

    xmin, xmax = float(x.min()), float(x.max())
    xr = max(xmax - xmin, 1.0)
    ax.set_xlim(xmin - x_pad_frac * xr, xmax + x_pad_frac * xr)

    ymin, ymax = float(y_np.min()), float(y_np.max())
    yr = ymax - ymin
    if yr < 1e-12:
        ax.set_ylim(ymin - 1e-6, ymax + 1e-6)
    else:
        pad = y_pad_frac * yr
        ax.set_ylim(ymin - pad, ymax + pad)

    out_path = str(out_path)
    os.makedirs(os.path.dirname(out_path) or ".", exist_ok=True)
    fig.savefig(out_path, bbox_inches="tight", pad_inches=0.02, transparent=transparent)
    plt.close(fig)


def main():
    p = argparse.ArgumentParser()
    p.add_argument("--debug_dir", type=str, required=True,
                   help="WAN entropy debug dir (e.g. outputs/entropy_scan_skate_24_32_v2/block_28)")
    p.add_argument("--which", type=str, default="final_ema", choices=["step", "final_ema"],
                   help="step = per-step curve from entropy_frame_step_XX.pt; final_ema = aggregated curve used for keyframe selection")
    p.add_argument("--step", type=int, default=4,
                   help="Step index when --which step (e.g. 4 for entropy_steps=5)")
    p.add_argument("--mode", type=str, default="ema",
                   help="When --which final_ema, mode for WAN2.1 filename: entropy_frame_final_{mode}.pt")
    p.add_argument("--out", type=str, required=True, help="Output path, e.g. .../entropy_curve_clean.png")
    p.add_argument("--dpi", type=int, default=300)
    p.add_argument("--linewidth", type=float, default=2.0)
    p.add_argument("--no_transparent", action="store_true", help="Disable transparent background")
    p.add_argument("--w", type=float, default=6.4, help="Figure width (inches)")
    p.add_argument("--h", type=float, default=4.8, help="Figure height (inches)")
    p.add_argument("--y_pad", type=float, default=0.06, help="Y-axis padding fraction (avoid curve touching top/bottom)")
    p.add_argument("--x_pad", type=float, default=0.02, help="X-axis padding fraction")
    args = p.parse_args()

    y, pt_path = _load_entropy_curve(args.debug_dir, args.which, args.step, args.mode)
    print(f"[OK] loaded: {pt_path}  shape={tuple(y.shape)}  min={float(y.min()):.6g} max={float(y.max()):.6g}")

    plot_clean_line(
        y=y,
        out_path=args.out,
        dpi=args.dpi,
        linewidth=args.linewidth,
        transparent=(not args.no_transparent),
        figsize=(args.w, args.h),
        y_pad_frac=args.y_pad,
        x_pad_frac=args.x_pad,
    )
    print(f"[OK] saved: {args.out}")


if __name__ == "__main__":
    main()
