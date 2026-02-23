#!/usr/bin/env python3
"""
测量「关键帧视频 → EDEN encoder 输出」的 latent 大小（HybridSD 式云→端传 latent 口径）。

三种口径：
- tensor_bytes: 纯张量占用（理论 latent 大小）
- packed_bytes: pack_enc_out() 后的 bytes（真实传输 payload）
- base64_bytes: 若走 HTTP/base64，实际字符串长度

用法示例：
  python tools/measure_eden_enc_latent_size.py \\
    --keyframes_mp4 outputs/cloud_keyframes_only/block_28/keyframes.mp4 \\
    --eden_config configs/eval_eden.yaml \\
    --device cuda:0 \\
    --fp16 \\
    --out_json outputs/cloud_keyframes_only/block_28/eden_enc_latent_size_fp16.json
"""
import argparse
import json
import os
import sys
from pathlib import Path

# 与 scripts/run_full_pipeline.py 一致：允许从项目根直接运行时 import vdit
_SCRIPT = Path(__file__).resolve()
for _p in [_SCRIPT] + list(_SCRIPT.parents):
    _src = _p / "src"
    if _src.is_dir():
        _root = _src.parent
        if str(_root) not in sys.path:
            sys.path.insert(0, str(_root))
        if str(_src) not in sys.path:
            sys.path.insert(0, str(_src))
        break

import torch
import yaml

from vdit.models import load_model
from vdit.pipeline.video_io import read_video_tensor
from vdit.utils import InputPadder
from vdit.utils.encode_transfer import pack_enc_out


def tensor_bytes(d: dict) -> int:
    total = 0
    for v in d.values():
        if torch.is_tensor(v):
            total += v.numel() * v.element_size()
    return total


def b64_size(n_bytes: int) -> int:
    """base64 编码后字节长度: 4 * ceil(n/3)"""
    return 4 * ((n_bytes + 2) // 3)


def main():
    ap = argparse.ArgumentParser(description="Measure EDEN encoder latent size for keyframe video (HybridSD-style).")
    ap.add_argument("--keyframes_mp4", required=True, help="Path to keyframes mp4 (e.g. cloud_keyframes_only/block_28/keyframes.mp4)")
    ap.add_argument("--eden_config", default="configs/eval_eden.yaml", help="EDEN config yaml")
    ap.add_argument("--device", default="cuda:0", help="Device for encoder")
    ap.add_argument("--fp16", action="store_true", help="Cast encoder outputs to fp16 before packing (transmission dtype)")
    ap.add_argument("--out_json", default=None, help="Optional: write full result to this json path")
    args = ap.parse_args()

    device = torch.device(args.device)

    # 1) Load config + EDEN
    with open(args.eden_config, "r", encoding="utf-8") as f:
        cfg = yaml.unsafe_load(f)
    eden = load_model(cfg["model_name"], **cfg["model_args"])
    ckpt = torch.load(cfg["pretrained_eden_path"], map_location="cpu")
    eden.load_state_dict(ckpt["eden"])
    eden.to(device).eval()

    # 2) Read keyframes video -> [T, 3, H, W] float [0,1]
    frames, video_info = read_video_tensor(args.keyframes_mp4)
    T, _, H, W = frames.shape

    per_pair = []
    total_tensor = 0
    total_packed = 0
    total_b64 = 0

    # 3) Encode each adjacent keyframe pair
    for i in range(T - 1):
        f0 = frames[i : i + 1].to(device)   # [1, 3, H, W]
        f1 = frames[i + 1 : i + 2].to(device)

        padder = InputPadder([H, W])
        cond_frames = padder.pad(torch.cat([f0, f1], dim=0))  # [2, 3, H', W']

        with torch.no_grad():
            enc_out = eden.encode(cond_frames)

        # Optional: fp16 for transmission
        if args.fp16:
            for k in ("cond_dit", "cond_dec", "stats_mean", "stats_std"):
                if k in enc_out and torch.is_tensor(enc_out[k]):
                    enc_out[k] = enc_out[k].half()

        # 4) Compute sizes
        sz_tensor = tensor_bytes(enc_out)
        blob = pack_enc_out(enc_out)
        sz_packed = len(blob)
        sz_b64 = b64_size(sz_packed)

        per_pair.append({
            "pair": [int(i), int(i + 1)],
            "tensor_bytes": sz_tensor,
            "packed_bytes": sz_packed,
            "base64_bytes": sz_b64,
        })
        total_tensor += sz_tensor
        total_packed += sz_packed
        total_b64 += sz_b64

    result = {
        "keyframes_mp4": args.keyframes_mp4,
        "num_keyframes": T,
        "num_pairs": T - 1,
        "fp16": bool(args.fp16),
        "totals": {
            "tensor_bytes": total_tensor,
            "packed_bytes": total_packed,
            "base64_bytes": total_b64,
        },
        "per_pair": per_pair,
        "video_info": {
            "height": video_info.height,
            "width": video_info.width,
            "fps": video_info.fps,
        },
    }

    print(json.dumps(result["totals"], indent=2))
    if args.out_json:
        os.makedirs(os.path.dirname(args.out_json) or ".", exist_ok=True)
        with open(args.out_json, "w", encoding="utf-8") as f:
            json.dump(result, f, indent=2, ensure_ascii=False)
        print(f"[OK] wrote: {args.out_json}")


if __name__ == "__main__":
    main()
