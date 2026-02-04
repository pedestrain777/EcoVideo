"""
统一入口脚本（当前：video->keyframes->RAFT打分->EDEN动态插帧->输出）。

示例：
python scripts/run_pipeline.py \
  --video_path examples/input.mp4 \
  --eden_config configs/eval_eden.yaml \
  --output_path interpolation_outputs/out.mp4 \
  --raft_ckpt /data/models/raft/raft-things.pth
"""

from __future__ import annotations

import argparse
import csv
import json
import os
import sys
import time
from datetime import datetime
from pathlib import Path

# 允许直接 `python scripts/run_pipeline.py ...` 时 import vdit.*
_HERE = Path(__file__).resolve()
for _parent in [_HERE] + list(_HERE.parents):
    _src_dir = _parent / "src"
    if _src_dir.is_dir():
        _root = _src_dir.parent
        if str(_root) not in sys.path:
            sys.path.insert(0, str(_root))
        if str(_src_dir) not in sys.path:
            sys.path.insert(0, str(_src_dir))
        break


def main() -> None:
    p = argparse.ArgumentParser()
    p.add_argument("--video_path", type=str, required=True)
    p.add_argument("--eden_config", type=str, required=True)
    p.add_argument("--output_path", type=str, default="interpolation_outputs/interpolated.mp4")
    p.add_argument("--log_file", type=str, default="interpolation_outputs/greedy_refinement.log")

    # RAFT 权重默认路径（和常见 RAFT demo 的参数习惯对齐）
    # 若你机器上权重在别处，运行时用 --raft_ckpt 覆盖即可。
    p.add_argument(
        "--raft_ckpt",
        type=str,
        default="/data/models/raft/raft-things.pth",
        help="restore checkpoint (default: /data/models/raft/raft-things.pth)",
    )
    p.add_argument("--raft_device", type=str, default="cuda:0")
    p.add_argument("--eden_device", type=str, default="cuda:0")
    p.add_argument("--use_split_gpu", action="store_true")

    p.add_argument("--target_fps", type=float, default=24.0)
    p.add_argument("--keyframe_mode", type=str, choices=["all", "uniform", "random"], default="all")
    p.add_argument("--keyframes_k", type=int, default=8)
    p.add_argument("--seed", type=int, default=0)
    p.add_argument("--topk_ratio", type=float, default=0.1)

    # -------- NEW: append metrics (JSONL/CSV) --------
    p.add_argument("--sample_id", type=str, default=None, help="Optional sample id for joining cloud/edge metrics.")
    p.add_argument("--metrics_jsonl", type=str, default=None, help="Append edge metrics as JSONL.")
    p.add_argument("--metrics_csv", type=str, default=None, help="Append edge metrics as CSV.")

    args = p.parse_args()

    os.makedirs(os.path.dirname(args.output_path) or ".", exist_ok=True)
    os.makedirs(os.path.dirname(args.log_file) or ".", exist_ok=True)
    if args.metrics_jsonl:
        os.makedirs(os.path.dirname(args.metrics_jsonl) or ".", exist_ok=True)
    if args.metrics_csv:
        os.makedirs(os.path.dirname(args.metrics_csv) or ".", exist_ok=True)

    # 延迟导入：避免 `--help` 也触发重依赖（如 lpips/xformers）导入失败
    from vdit.pipeline.run_iframe import PipelineConfig, run_interpolation_pipeline

    cfg = PipelineConfig(
        eden_config=args.eden_config,
        raft_ckpt=args.raft_ckpt,
        raft_device=args.raft_device,
        eden_device=args.eden_device,
        use_split_gpu=args.use_split_gpu,
        target_fps=args.target_fps,
        keyframe_mode=args.keyframe_mode,
        keyframes_k=args.keyframes_k,
        seed=args.seed,
        topk_ratio=args.topk_ratio,
    )

    t0 = time.perf_counter()
    run_interpolation_pipeline(
        video_path=args.video_path,
        output_path=args.output_path,
        cfg=cfg,
        log_file=args.log_file,
    )
    edge_latency = float(time.perf_counter() - t0)

    # sample_id default: stem of input keyframes video
    if args.sample_id:
        sid = args.sample_id
    else:
        sid = Path(args.video_path).stem

    record = {
        "ts": datetime.utcnow().isoformat(timespec="seconds") + "Z",
        "role": "edge",
        "sample_id": sid,
        "timing": {"edge_latency_sec": edge_latency},
        "io": {"input_keyframes_video": args.video_path, "output_video": args.output_path},
        "cfg": {
            "eden_config": args.eden_config,
            "target_fps": float(args.target_fps),
            "keyframe_mode": args.keyframe_mode,
            "use_split_gpu": bool(args.use_split_gpu),
            "raft_ckpt": args.raft_ckpt,
            "topk_ratio": float(args.topk_ratio),
        },
    }

    # append JSONL
    if args.metrics_jsonl:
        with open(args.metrics_jsonl, "a", encoding="utf-8") as f:
            f.write(json.dumps(record, ensure_ascii=False) + "\n")

    # append CSV
    if args.metrics_csv:
        csv_fields = [
            "ts", "role", "sample_id",
            "edge_latency_sec",
            "target_fps", "keyframe_mode", "use_split_gpu", "topk_ratio",
            "raft_ckpt",
            "input_keyframes_video", "output_video",
            "eden_config",
        ]
        row = {
            "ts": record["ts"],
            "role": "edge",
            "sample_id": sid,
            "edge_latency_sec": edge_latency,
            "target_fps": record["cfg"]["target_fps"],
            "keyframe_mode": record["cfg"]["keyframe_mode"],
            "use_split_gpu": record["cfg"]["use_split_gpu"],
            "topk_ratio": record["cfg"]["topk_ratio"],
            "raft_ckpt": record["cfg"]["raft_ckpt"],
            "input_keyframes_video": args.video_path,
            "output_video": args.output_path,
            "eden_config": args.eden_config,
        }
        file_exists = os.path.isfile(args.metrics_csv)
        with open(args.metrics_csv, "a", encoding="utf-8", newline="") as f:
            w = csv.DictWriter(f, fieldnames=csv_fields)
            if not file_exists:
                w.writeheader()
            w.writerow(row)


if __name__ == "__main__":
    main()

