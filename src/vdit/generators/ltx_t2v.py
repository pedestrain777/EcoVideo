from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
from typing import Optional, Tuple
import sys
import math
import os

import torch
import yaml

from vdit.generators.base import register_generator


# VDiT vendors LTX under: third_party/ltx_video/ltx_video
try:
    import ltx_video  # type: ignore
except Exception:
    _ROOT = Path(__file__).resolve().parents[3]
    _LTX_ROOT = _ROOT / "third_party" / "ltx_video"
    if _LTX_ROOT.exists():
        sys.path.insert(0, str(_LTX_ROOT))
    import ltx_video  # type: ignore

from ltx_video.inference import create_ltx_video_pipeline  # type: ignore
from ltx_video.pipelines.pipeline_ltx_video import LTXMultiScalePipeline  # type: ignore
from ltx_video.models.autoencoders.latent_upsampler import LatentUpsampler  # type: ignore
from ltx_video.utils.skip_layer_strategy import SkipLayerStrategy  # type: ignore


@dataclass(frozen=True)
class LtxGenerateConfig:
    # ---- model / runtime ----
    precision: str = "bfloat16"  # "bfloat16" | "float8_e4m3fn" | others supported by LTX
    sampler: Optional[str] = None  # None / "from_checkpoint" / "uniform" / "linear-quadratic"
    device: str = "cuda"
    seed: int = 0

    # ---- video / sampling ----
    height: int = 512
    width: int = 768
    num_frames: int = 81
    frame_rate: float = 24.0
    num_inference_steps: int = 50

    guidance_scale: float = 5.0
    stg_scale: float = 0.0
    rescaling_scale: float = 0.7
    cfg_star_rescale: bool = False
    mixed_precision: bool = False
    offload_to_cpu: bool = False
    stochastic_sampling: bool = False

    # ---- WAN-style keyframe-by-entropy ----
    keyframe_by_entropy: bool = False
    entropy_steps: int = 5
    entropy_mode: str = "ema"  # {"ema","mean","last"}
    entropy_ema_alpha: float = 0.6
    entropy_block_idx: int = -1
    keyframe_topk: int = 16
    keyframe_cover: bool = True
    use_nonkey_context: bool = True

    # effective fps after prune (optional override)
    keyframe_out_fps: Optional[float] = None
    keyframe_target_fps: Optional[float] = None

    # ---- nonkey_update_mode: none / interval / teacache ----
    nonkey_update_mode: str = "none"
    nonkey_update_interval: int = 5
    teacache_rel_l1_thresh: float = 0.02
    teacache_max_skip: int = 8
    teacache_warmup: int = 2

    # ---- multi-scale (official pipeline yaml) ----
    pipeline_config_path: Optional[str] = None  # official yaml path
    spatial_upscaler_ckpt: Optional[str] = None  # override yaml spatial_upscaler_model_path
    downscale_factor: Optional[float] = None  # override yaml downscale_factor


def auto_keyframe_topk_ltx(
    frame_num_full: int,
    fps_full: float,
    fps_key: float,
    stride_t: int,
    min_k: int = 2,
    max_k: Optional[int] = None,
) -> int:
    """
    LTX 版本：根据目标关键帧视频 fps 自动估计 latent 关键帧数量 K。
    近似关系：pixel_frames ≈ latent_frames * stride_t
    """
    if fps_key is None:
        raise ValueError("fps_key is None")
    if fps_key <= 0:
        raise ValueError(f"fps_key must be > 0, got {fps_key}")
    if fps_full <= 0:
        raise ValueError(f"fps_full must be > 0, got {fps_full}")
    if frame_num_full <= 1:
        return max(min_k, 1)
    if stride_t <= 0:
        raise ValueError(f"stride_t must be > 0, got {stride_t}")

    # 目标关键帧视频想保留的像素帧数（近似保持时长一致）
    t_key = int(round(frame_num_full * (fps_key / float(fps_full))))
    t_key = max(1, t_key)

    # LTX: pixel_frames ≈ latent_frames * stride_t
    k = int(math.ceil(t_key / float(stride_t)))
    k = max(min_k, k)

    if max_k is not None:
        k = min(max_k, k)

    return k


def _ltx_video_to_vdit_frames(video_bcfhw: torch.Tensor) -> torch.Tensor:
    """
    LTX:  [B,C,F,H,W] float, value range typically [-1,1] or [0,1]
    VDiT: [T,3,H,W] float in [0,1] on CPU
    """
    if video_bcfhw.ndim != 5:
        raise ValueError(f"Expect LTX video [B,C,F,H,W], got {tuple(video_bcfhw.shape)}")
    b, c, f, h, w = video_bcfhw.shape
    if b != 1:
        video_bcfhw = video_bcfhw[:1]
    if c != 3:
        raise ValueError(f"Expect C=3 RGB video, got C={c}")

    v = video_bcfhw[0]  # [C,F,H,W]
    # map [-1,1] -> [0,1] if needed
    if v.min().item() < -0.1:
        v = (v.clamp(-1.0, 1.0) + 1.0) * 0.5
    else:
        v = v.clamp(0.0, 1.0)

    frames = v.permute(1, 0, 2, 3).contiguous()  # [F,3,H,W]
    return frames.float().cpu()


def _pad_to_multiple(x: int, m: int) -> int:
    return ((x - 1) // m + 1) * m


def _pad_num_frames_ltx(num_frames: int, video_scale_factor: int) -> int:
    # official: (N*8 + 1) style; generalized to video_scale_factor
    if num_frames <= 1:
        return 1
    return ((num_frames - 2) // video_scale_factor + 1) * video_scale_factor + 1


def _calc_center_padding(
    src_h: int, src_w: int, tgt_h: int, tgt_w: int
) -> Tuple[int, int, int, int]:
    # returns (pad_left, pad_right, pad_top, pad_bottom) as in official inference
    pad_h = tgt_h - src_h
    pad_w = tgt_w - src_w
    pad_top = pad_h // 2
    pad_bottom = pad_h - pad_top
    pad_left = pad_w // 2
    pad_right = pad_w - pad_left
    return (pad_left, pad_right, pad_top, pad_bottom)


def _stg_mode_to_strategy(stg_mode: Optional[str]) -> Optional[SkipLayerStrategy]:
    if not stg_mode:
        return None
    s = stg_mode.lower()
    if s in ("stg_av", "attention_values"):
        return SkipLayerStrategy.AttentionValues
    if s in ("stg_as", "attention_skip"):
        return SkipLayerStrategy.AttentionSkip
    if s in ("stg_r", "residual"):
        return SkipLayerStrategy.Residual
    if s in ("stg_t", "transformer_block"):
        return SkipLayerStrategy.TransformerBlock
    raise ValueError(f"Invalid stg_mode: {stg_mode}")


def _load_pipeline_yaml(path: str) -> dict:
    with open(path, "r") as f:
        obj = yaml.safe_load(f)
    if not isinstance(obj, dict):
        raise ValueError(f"pipeline yaml must be a dict, got {type(obj)}")
    return obj


@torch.no_grad()
def generate_ltx_frames(
    *,
    prompt: str,
    ckpt_path: str,
    text_encoder_path: str,
    cfg: LtxGenerateConfig,
) -> Tuple[torch.Tensor, float]:
    pipe = create_ltx_video_pipeline(
        ckpt_path=ckpt_path,
        precision=cfg.precision,
        text_encoder_model_name_or_path=text_encoder_path,
        sampler=cfg.sampler,
        device=cfg.device,
        enhance_prompt=False,
    )

    # ---- Optional: load official pipeline yaml (enables multi-scale) ----
    pipeline_cfg: Optional[dict] = None
    pipeline_type = None
    skip_layer_strategy: Optional[SkipLayerStrategy] = None

    if cfg.pipeline_config_path:
        pipeline_cfg = _load_pipeline_yaml(cfg.pipeline_config_path)
        pipeline_type = pipeline_cfg.get("pipeline_type", None)
        skip_layer_strategy = _stg_mode_to_strategy(pipeline_cfg.get("stg_mode", None))

    # ---- WAN-style: keyframe_target_fps -> auto keyframe_topk ----
    keyframe_topk = int(cfg.keyframe_topk)
    keyframe_out_fps = cfg.keyframe_out_fps

    if cfg.keyframe_by_entropy and cfg.keyframe_target_fps is not None:
        from ltx_video.models.autoencoders.vae_encode import (  # type: ignore
            get_vae_size_scale_factor,
        )

        stride_t = int(get_vae_size_scale_factor(pipe.vae)[0])
        max_k = int(math.ceil(int(cfg.num_frames) / float(stride_t)))

        keyframe_topk = auto_keyframe_topk_ltx(
            frame_num_full=int(cfg.num_frames),
            fps_full=float(cfg.frame_rate),
            fps_key=float(cfg.keyframe_target_fps),
            stride_t=stride_t,
            min_k=2,
            max_k=max_k,
        )
        if keyframe_out_fps is None:
            keyframe_out_fps = float(cfg.keyframe_target_fps)

    g = torch.Generator(device=cfg.device).manual_seed(int(cfg.seed))

    # ---- Official-style padding: H/W divisible by 32, frames -> (N*scale+1) ----
    if int(cfg.num_frames) < 1:
        raise ValueError(f"cfg.num_frames must be >= 1, got {cfg.num_frames}")

    video_scale_factor = int(getattr(pipe, "video_scale_factor", 8))
    if video_scale_factor <= 0:
        video_scale_factor = 8

    height_padded = _pad_to_multiple(int(cfg.height), 32)
    width_padded = _pad_to_multiple(int(cfg.width), 32)
    num_frames_padded = _pad_num_frames_ltx(int(cfg.num_frames), video_scale_factor)

    pad_left, pad_right, pad_top, pad_bottom = _calc_center_padding(
        int(cfg.height), int(cfg.width), height_padded, width_padded
    )

    negative_prompt = "worst quality, inconsistent motion, blurry, jittery, distorted"

    base_kwargs = dict(
        height=int(height_padded),
        width=int(width_padded),
        num_frames=int(num_frames_padded),
        frame_rate=float(cfg.frame_rate),
        prompt=prompt,
        negative_prompt=negative_prompt,
        generator=g,
        output_type="pt",
        return_dict=True,
        is_video=True,
        vae_per_channel_normalize=True,
        image_cond_noise_scale=0.0,
        mixed_precision=bool(cfg.mixed_precision),
        offload_to_cpu=bool(cfg.offload_to_cpu),
        stochastic_sampling=bool(cfg.stochastic_sampling),
    )

    # ---- pipeline_config kwargs (decode_timestep/noise, first_pass/second_pass, etc.) ----
    call_cfg_kwargs: dict = {}
    if pipeline_cfg:
        call_cfg_kwargs.update(pipeline_cfg)
        call_cfg_kwargs.pop("stg_mode", None)
        call_cfg_kwargs.pop("checkpoint_path", None)
        call_cfg_kwargs.pop("text_encoder_model_name_or_path", None)
        call_cfg_kwargs.pop("precision", None)
        call_cfg_kwargs.pop("sampler", None)
        call_cfg_kwargs.pop("prompt_enhancer_image_caption_model_name_or_path", None)
        call_cfg_kwargs.pop("prompt_enhancer_llm_model_name_or_path", None)
        call_cfg_kwargs.pop("prompt_enhancement_words_threshold", None)
        call_cfg_kwargs.pop("spatial_upscaler_model_path", None)
        # Avoid passing duplicated kwargs that we already set via base_kwargs or explicit args.
        # These may appear in the yaml and would otherwise cause "multiple values for keyword argument".
        call_cfg_kwargs.pop("downscale_factor", None)
        call_cfg_kwargs.pop("stochastic_sampling", None)
        # multi-scale pipeline also defines these; we control them explicitly
        call_cfg_kwargs.pop("first_pass", None)
        call_cfg_kwargs.pop("second_pass", None)

    # ---- single-stage path (default) ----
    if pipeline_type != "multi-scale":
        out = pipe(
            **base_kwargs,
            num_inference_steps=int(cfg.num_inference_steps),
            guidance_scale=float(cfg.guidance_scale),
            stg_scale=float(cfg.stg_scale),
            rescaling_scale=float(cfg.rescaling_scale),
            cfg_star_rescale=bool(cfg.cfg_star_rescale),
            skip_layer_strategy=skip_layer_strategy,
            keyframe_by_entropy=bool(cfg.keyframe_by_entropy),
            entropy_steps=int(cfg.entropy_steps),
            entropy_mode=str(cfg.entropy_mode),
            entropy_ema_alpha=float(cfg.entropy_ema_alpha),
            entropy_block_idx=int(cfg.entropy_block_idx),
            keyframe_topk=int(keyframe_topk),
            keyframe_cover=bool(cfg.keyframe_cover),
            use_nonkey_context=bool(cfg.use_nonkey_context),
            nonkey_update_mode=str(cfg.nonkey_update_mode),
            nonkey_update_interval=int(cfg.nonkey_update_interval),
            teacache_rel_l1_thresh=float(cfg.teacache_rel_l1_thresh),
            teacache_max_skip=int(cfg.teacache_max_skip),
            teacache_warmup=int(cfg.teacache_warmup),
            **call_cfg_kwargs,
        )
    else:
        # ---- multi-scale: wrap pipeline + inject entropy only into first_pass ----
        downscale_factor = float(
            pipeline_cfg.get("downscale_factor", 0.6666666)
        ) if pipeline_cfg else 0.6666666
        if cfg.downscale_factor is not None:
            downscale_factor = float(cfg.downscale_factor)

        up_path = None
        if cfg.spatial_upscaler_ckpt:
            up_path = cfg.spatial_upscaler_ckpt
        else:
            rel = (pipeline_cfg or {}).get("spatial_upscaler_model_path", None)
            if not rel:
                raise ValueError(
                    "multi-scale requires spatial upscaler ckpt. "
                    "Provide --ltx_spatial_upscaler_ckpt or set spatial_upscaler_model_path in yaml."
                )
            ydir = Path(cfg.pipeline_config_path).resolve().parent  # type: ignore[arg-type]
            rel_str = str(rel)
            up_path = (
                str((ydir / rel_str).resolve())
                if not os.path.isabs(rel_str)
                else rel_str
            )

        latent_upsampler = LatentUpsampler.from_pretrained(up_path).to(cfg.device).eval()
        pipe_ms = LTXMultiScalePipeline(pipe, latent_upsampler=latent_upsampler)

        first_pass = dict((pipeline_cfg or {}).get("first_pass", {}) or {})
        second_pass = dict((pipeline_cfg or {}).get("second_pass", {}) or {})

        if bool(cfg.keyframe_by_entropy):
            first_pass.update(
                dict(
                    keyframe_by_entropy=True,
                    entropy_steps=int(cfg.entropy_steps),
                    entropy_mode=str(cfg.entropy_mode),
                    entropy_ema_alpha=float(cfg.entropy_ema_alpha),
                    entropy_block_idx=int(cfg.entropy_block_idx),
                    keyframe_topk=int(keyframe_topk),
                    keyframe_cover=bool(cfg.keyframe_cover),
                    use_nonkey_context=bool(cfg.use_nonkey_context),
                    nonkey_update_mode=str(cfg.nonkey_update_mode),
                    nonkey_update_interval=int(cfg.nonkey_update_interval),
                    teacache_rel_l1_thresh=float(cfg.teacache_rel_l1_thresh),
                    teacache_max_skip=int(cfg.teacache_max_skip),
                    teacache_warmup=int(cfg.teacache_warmup),
                )
            )
            second_pass.update(
                dict(
                    keyframe_by_entropy=False,
                    nonkey_update_mode="none",
                    use_nonkey_context=False,
                )
            )

        out = pipe_ms(
            downscale_factor=downscale_factor,
            first_pass=first_pass,
            second_pass=second_pass,
            **base_kwargs,
            skip_layer_strategy=skip_layer_strategy,
            **call_cfg_kwargs,
        )

    video = out.images  # [B,C,F,H,W]

    # ---- crop back to requested frames and spatial size ----
    if video.ndim == 5:
        video = video[:, :, : int(cfg.num_frames), :, :]

        if (height_padded != int(cfg.height)) or (width_padded != int(cfg.width)):
            h0 = pad_top
            h1 = height_padded - pad_bottom
            w0 = pad_left
            w1 = width_padded - pad_right
            video = video[:, :, :, h0:h1, w0:w1]

    frames = _ltx_video_to_vdit_frames(video)

    fps_src = float(cfg.frame_rate)
    fps_tgt = fps_src
    if cfg.keyframe_by_entropy:
        t_out = int(frames.shape[0])
        t_full = int(cfg.num_frames)
        if t_full > 0:
            fps_tgt = fps_src * (t_out / float(t_full))
        if keyframe_out_fps is not None:
            fps_tgt = float(keyframe_out_fps)

    # Simple debug print to help sanity-check num_frames/fps behavior
    try:
        print(
            f"[LTX] out_frames={int(frames.shape[0])} "
            f"fps={float(fps_tgt)} "
            f"num_frames_cfg={int(cfg.num_frames)} "
            f"frame_rate_cfg={float(cfg.frame_rate)}",
            flush=True,
        )
    except Exception:
        pass

    del video
    try:
        torch.cuda.empty_cache()
    except Exception:
        pass

    return frames, float(fps_tgt)


@register_generator("ltx")
class LtxGenerator:
    def __init__(self, ckpt_path: str, text_encoder_path: str, cfg: LtxGenerateConfig):
        if not ckpt_path:
            raise ValueError("LTX requires ckpt_path (passed from --ckpt).")
        if not text_encoder_path:
            raise ValueError("LTX requires text_encoder_path (passed from --text_encoder_path).")
        self.ckpt_path = ckpt_path
        self.text_encoder_path = text_encoder_path
        self.cfg = cfg

    @torch.no_grad()
    def generate(self, prompt: str) -> Tuple[torch.Tensor, float]:
        return generate_ltx_frames(
            prompt=prompt,
            ckpt_path=self.ckpt_path,
            text_encoder_path=self.text_encoder_path,
            cfg=self.cfg,
        )


