# scripts/run_full_pipeline.py

from __future__ import annotations

import argparse
import os
import sys
from pathlib import Path

# 允许直接 `python scripts/run_full_pipeline.py ...` 时 import vdit.*
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

    # -------- WAN 输入 --------
    p.add_argument("--wan_ckpt_dir", type=str, default=None, help="WAN checkpoint directory (required if not using --input_video)")
    p.add_argument("--prompt", type=str, default=None, help="Text prompt for WAN (required if not using --input_video)")
    
    # 可选：直接从已有视频开始（跳过 WAN 生成，节省时间）
    p.add_argument("--input_video", type=str, default=None, help="Input video path (skip WAN generation if provided)")
    p.add_argument("--input_fps", type=float, default=None, help="Input video fps (auto-detect if not provided)")

    # WAN 生成参数（baseline：t2v-1.3B）
    p.add_argument("--wan_task", type=str, default="t2v-1.3B")
    p.add_argument("--wan_size", type=str, default="832*480", choices=["832*480", "480*832"])
    p.add_argument("--wan_frame_num", type=int, default=81)
    p.add_argument("--wan_sample_solver", type=str, default="unipc", choices=["unipc", "dpm++"])
    p.add_argument("--wan_sample_steps", type=int, default=50)
    p.add_argument("--wan_sample_shift", type=float, default=5.0)
    p.add_argument("--wan_guide_scale", type=float, default=5.0)
    p.add_argument("--wan_seed", type=int, default=0)
    p.add_argument("--wan_offload_model", action="store_true", help="enable offload_model=True (recommended)")

    # WAN 内部“均匀/随机取帧”（按 fps 下采样，保持时长不变）
    p.add_argument("--wan_out_fps", type=float, default=None, help="e.g. 8 or 12; None means no downsample")
    p.add_argument(
        "--wan_frame_sample",
        type=str,
        default="uniform",
        choices=["uniform", "random", "stratified_random"],
    )
    p.add_argument("--wan_frame_sample_seed", type=int, default=0)
    p.add_argument("--generator", type=str, default="wan", help="generator backend name (default: wan)")

    # -------- WAN: entropy keyframe（真正裁剪 latent 时间维）--------
    p.add_argument("--wan_keyframe_by_entropy", action="store_true")
    p.add_argument("--wan_entropy_steps", type=int, default=5)
    p.add_argument("--wan_entropy_mode", type=str, default="mean", choices=["last", "mean", "ema"])
    p.add_argument("--wan_entropy_ema_alpha", type=float, default=0.6)
    p.add_argument("--wan_entropy_block_idx", type=int, default=-1)
    p.add_argument("--wan_keyframe_topk", type=int, default=16)
    p.add_argument("--wan_keyframe_cover", action="store_true")
    p.add_argument("--wan_no_keyframe_cover", action="store_true")
    p.add_argument("--wan_use_nonkey_context", action="store_true")
    p.add_argument("--wan_no_nonkey_context", action="store_true")
    p.add_argument("--wan_entropy_debug_dir", type=str, default=None)
    p.add_argument("--wan_save_debug_pt", action="store_true")
    p.add_argument("--wan_no_save_debug_pt", action="store_true")
    p.add_argument("--wan_profile_timing", action="store_true")
    p.add_argument("--wan_no_profile_timing", action="store_true")
    p.add_argument("--wan_keyframe_out_fps", type=float, default=None)
    p.add_argument("--wan_keyframe_target_fps", type=float, default=None)

    # -------- WAN: method-2 non-key low-frequency compute --------
    p.add_argument("--wan_nonkey_update_mode",
                   type=str,
                   default="none",
                   choices=["none", "interval", "teacache"])
    p.add_argument("--wan_nonkey_update_interval", type=int, default=5)
    p.add_argument("--wan_teacache_rel_l1_thresh", type=float, default=0.02)
    p.add_argument("--wan_teacache_max_skip", type=int, default=10)
    p.add_argument("--wan_teacache_warmup", type=int, default=0)
    p.add_argument("--wan_save_teacache_trace_png", action="store_true")
    p.add_argument("--wan_no_save_teacache_trace_png", action="store_true")

    # -------- 插帧参数（你原来的 pipeline 参数）--------
    p.add_argument("--eden_config", type=str, required=True)
    p.add_argument("--output_path", type=str, default="interpolation_outputs/final.mp4")
    p.add_argument("--log_file", type=str, default="interpolation_outputs/greedy_refinement.log")
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

    # 关键帧策略：如果你用了 WAN out_fps 做“先取帧”，建议这里用 all（避免二次采样）
    p.add_argument("--keyframe_mode", type=str, choices=["all", "uniform", "random"], default="all")
    p.add_argument("--keyframes_k", type=int, default=8)
    p.add_argument("--seed", type=int, default=0)
    p.add_argument("--topk_ratio", type=float, default=0.1)

    # -------- NEW: 可选 baseline（WAN 生成完整视频）用于对比 --------
    p.add_argument(
        "--wan_generate_full_baseline",
        action="store_true",
        help="Also generate a full WAN video (same prompt/seed/steps/solver) for comparison.",
    )
    p.add_argument(
        "--save_wan_full_baseline_video",
        type=str,
        default=None,
        help="If set, save the baseline full WAN video to this path.",
    )
    p.add_argument(
        "--metrics_json",
        type=str,
        default=None,
        help="If set, save timing/speedup metrics to this json path. Default: <output>.metrics.json",
    )

    # -------- NEW: cloud-only mode (WAN -> keyframes.mp4, then exit) --------
    p.add_argument(
        "--stop_after_wan",
        action="store_true",
        help="If set: only run WAN generation and save keyframes video, then exit (no EDEN interpolation).",
    )
    p.add_argument(
        "--save_keyframes_video",
        type=str,
        default=None,
        help="Cloud output: keyframes video path (mp4). Required if --stop_after_wan is set.",
    )

    # -------- NEW: append metrics (JSONL/CSV) --------
    p.add_argument("--sample_id", type=str, default=None, help="Optional sample id for joining cloud/edge metrics.")
    p.add_argument(
        "--cloud_metrics_jsonl",
        type=str,
        default=None,
        help="Append cloud metrics as JSONL (one line per sample).",
    )
    p.add_argument(
        "--cloud_metrics_csv",
        type=str,
        default=None,
        help="Append cloud metrics as CSV (one row per sample).",
    )

    # 可选：保存 WAN 中间视频
    p.add_argument("--save_wan_video", type=str, default=None, help="save WAN raw/downsampled video for debugging")
    p.add_argument(
        "--save_sampled_video",
        type=str,
        default=None,
        help="save sampled video (after uniform/random sampling stage)",
    )

    args = p.parse_args()

    # 参数验证
    if args.input_video is None and (args.wan_ckpt_dir is None or args.prompt is None):
        p.error("Must provide either --input_video or (--wan_ckpt_dir + --prompt)")
    if args.stop_after_wan and not args.save_keyframes_video:
        p.error("--stop_after_wan requires --save_keyframes_video")

    os.makedirs(os.path.dirname(args.output_path) or ".", exist_ok=True)
    os.makedirs(os.path.dirname(args.log_file) or ".", exist_ok=True)
    if args.save_wan_video:
        os.makedirs(os.path.dirname(args.save_wan_video) or ".", exist_ok=True)
    if args.save_sampled_video:
        os.makedirs(os.path.dirname(args.save_sampled_video) or ".", exist_ok=True)
    if args.save_wan_full_baseline_video:
        os.makedirs(os.path.dirname(args.save_wan_full_baseline_video) or ".", exist_ok=True)
    if args.metrics_json:
        os.makedirs(os.path.dirname(args.metrics_json) or ".", exist_ok=True)
    if args.save_keyframes_video:
        os.makedirs(os.path.dirname(args.save_keyframes_video) or ".", exist_ok=True)
    if args.cloud_metrics_jsonl:
        os.makedirs(os.path.dirname(args.cloud_metrics_jsonl) or ".", exist_ok=True)
    if args.cloud_metrics_csv:
        os.makedirs(os.path.dirname(args.cloud_metrics_csv) or ".", exist_ok=True)
    if args.wan_entropy_debug_dir:
        os.makedirs(args.wan_entropy_debug_dir, exist_ok=True)

    # 延迟导入（与 run_pipeline.py 一致）
    from vdit.generators.wan_t2v import WanGenerateConfig
    from vdit.pipeline.full_pipeline import FullPipelineConfig, run_full_pipeline
    from vdit.pipeline.run_iframe import PipelineConfig

    # WAN 配置（仅在需要 WAN 生成时使用）
    keyframe_cover = True
    if args.wan_no_keyframe_cover:
        keyframe_cover = False
    elif args.wan_keyframe_cover:
        keyframe_cover = True

    use_nonkey_context = True
    if args.wan_no_nonkey_context:
        use_nonkey_context = False
    elif args.wan_use_nonkey_context:
        use_nonkey_context = True

    save_debug_pt = True
    if args.wan_no_save_debug_pt:
        save_debug_pt = False
    elif args.wan_save_debug_pt:
        save_debug_pt = True

    profile_timing = True
    if args.wan_no_profile_timing:
        profile_timing = False
    elif args.wan_profile_timing:
        profile_timing = True

    save_teacache_trace_png = True
    if args.wan_no_save_teacache_trace_png:
        save_teacache_trace_png = False
    elif args.wan_save_teacache_trace_png:
        save_teacache_trace_png = True

    wan_cfg = WanGenerateConfig(
        task=args.wan_task,
        size=args.wan_size,
        frame_num=args.wan_frame_num,
        sample_solver=args.wan_sample_solver,
        sample_steps=args.wan_sample_steps,
        sample_shift=args.wan_sample_shift,
        guide_scale=args.wan_guide_scale,
        seed=args.wan_seed,
        offload_model=(True if args.wan_offload_model else True),  # 默认 True
        out_fps=args.wan_out_fps,
        frame_sample=args.wan_frame_sample,
        frame_sample_seed=args.wan_frame_sample_seed,
        keyframe_by_entropy=args.wan_keyframe_by_entropy,
        entropy_steps=args.wan_entropy_steps,
        entropy_mode=args.wan_entropy_mode,
        entropy_ema_alpha=args.wan_entropy_ema_alpha,
        entropy_block_idx=args.wan_entropy_block_idx,
        keyframe_topk=args.wan_keyframe_topk,
        keyframe_cover=keyframe_cover,
        use_nonkey_context=use_nonkey_context,
        debug_dir=args.wan_entropy_debug_dir,
        save_debug_pt=save_debug_pt,
        profile_timing=profile_timing,
        keyframe_out_fps=args.wan_keyframe_out_fps,
        keyframe_target_fps=args.wan_keyframe_target_fps,
        nonkey_update_mode=args.wan_nonkey_update_mode,
        nonkey_update_interval=args.wan_nonkey_update_interval,
        teacache_rel_l1_thresh=args.wan_teacache_rel_l1_thresh,
        teacache_max_skip=args.wan_teacache_max_skip,
        teacache_warmup=args.wan_teacache_warmup,
        save_teacache_trace_png=save_teacache_trace_png,
        device_id=0,
        t5_cpu=False,
    )

    iframe_cfg = PipelineConfig(
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

    full_cfg = FullPipelineConfig(
        wan=wan_cfg,
        iframe=iframe_cfg,
        generator_name=args.generator,
        stop_after_wan=args.stop_after_wan,
        save_keyframes_video_path=args.save_keyframes_video,
        sample_id=args.sample_id,
        cloud_metrics_jsonl_path=args.cloud_metrics_jsonl,
        cloud_metrics_csv_path=args.cloud_metrics_csv,
    )

    run_full_pipeline(
        prompt=args.prompt,
        wan_ckpt_dir=args.wan_ckpt_dir,
        input_video=args.input_video,
        input_fps=args.input_fps,
        output_path=args.output_path,
        cfg=full_cfg,
        log_file=args.log_file,
        save_sampled_video_path=args.save_sampled_video,
        save_keyframes_video_path=args.save_keyframes_video,  # 仍保留：用于"单机插帧流程"时保存 preview
        save_wan_video_path=args.save_wan_video,
        generate_wan_full_baseline=args.wan_generate_full_baseline,
        save_wan_full_baseline_video_path=args.save_wan_full_baseline_video,
        metrics_json_path=args.metrics_json,
    )


if __name__ == "__main__":
    main()
