from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
from typing import Optional, Tuple
import sys

import torch

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

    # ---- nonkey_update_mode: none / interval / teacache ----
    nonkey_update_mode: str = "none"
    nonkey_update_interval: int = 5
    teacache_rel_l1_thresh: float = 0.02
    teacache_max_skip: int = 8
    teacache_warmup: int = 2


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

    g = torch.Generator(device=cfg.device).manual_seed(int(cfg.seed))

    out = pipe(
        height=cfg.height,
        width=cfg.width,
        num_frames=cfg.num_frames,
        frame_rate=float(cfg.frame_rate),
        prompt=prompt,
        negative_prompt="",
        num_inference_steps=int(cfg.num_inference_steps),
        guidance_scale=float(cfg.guidance_scale),
        stg_scale=float(cfg.stg_scale),
        rescaling_scale=float(cfg.rescaling_scale),
        cfg_star_rescale=bool(cfg.cfg_star_rescale),
        mixed_precision=bool(cfg.mixed_precision),
        offload_to_cpu=bool(cfg.offload_to_cpu),
        stochastic_sampling=bool(cfg.stochastic_sampling),
        generator=g,
        output_type="pt",
        return_dict=True,
        # WAN-style prune + nonkey update
        keyframe_by_entropy=bool(cfg.keyframe_by_entropy),
        entropy_steps=int(cfg.entropy_steps),
        entropy_mode=str(cfg.entropy_mode),
        entropy_ema_alpha=float(cfg.entropy_ema_alpha),
        entropy_block_idx=int(cfg.entropy_block_idx),
        keyframe_topk=int(cfg.keyframe_topk),
        keyframe_cover=bool(cfg.keyframe_cover),
        use_nonkey_context=bool(cfg.use_nonkey_context),
        nonkey_update_mode=str(cfg.nonkey_update_mode),
        nonkey_update_interval=int(cfg.nonkey_update_interval),
        teacache_rel_l1_thresh=float(cfg.teacache_rel_l1_thresh),
        teacache_max_skip=int(cfg.teacache_max_skip),
        teacache_warmup=int(cfg.teacache_warmup),
    )

    video = out.images  # [B,C,F,H,W]
    frames = _ltx_video_to_vdit_frames(video)

    fps_src = float(cfg.frame_rate)
    fps_tgt = fps_src
    if cfg.keyframe_by_entropy:
        t_out = int(frames.shape[0])
        t_full = int(cfg.num_frames)
        if t_full > 0:
            fps_tgt = fps_src * (t_out / float(t_full))
        if cfg.keyframe_out_fps is not None:
            fps_tgt = float(cfg.keyframe_out_fps)

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


