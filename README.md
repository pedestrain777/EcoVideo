# VDiT：生成 + 关键帧采样 + 自适应插帧（WAN → EDEN）

本仓库实现一条可扩展的视频处理 baseline pipeline：

> **生成器（默认 WAN 1.3B） → 生成视频 → 按 fps 均匀/随机取帧（关键帧序列，可保存中间视频） →
> 关键帧区间信息计算 → 自适应决定每段插帧数量（greedy refine） → EDEN 插帧 → 输出 24fps 完整视频**

本项目已工程化支持：

* 从 **prompt** 直接生成并插帧输出（WAN → EDEN）
* 从 **已有视频** 直接做插帧输出（跳过 WAN）
* 保存中间结果（采样后视频、关键帧预览视频、最终输出）
* xFormers 不可用时自动 fallback 到 PyTorch SDPA（保证可跑通）
* 使用 PyAV 写 mp4（避免 torchvision+PyAV 版本不兼容问题）

---

## 1. 项目结构

```
VDiT/
  README.md
  requirements.txt
  scripts/
    run_full_pipeline.py        # WAN->采样->插帧->输出（主入口）
    run_pipeline.py             # 从已有视频插帧（旧入口，仍可用）
  configs/
    eval_eden.yaml              # EDEN/插帧评估相关配置（也可用于推理）
  src/
    vdit/
      generators/               # 生成器插件（当前实现 wan）
      pipeline/                 # full_pipeline + run_iframe + video_io
      interpolators/            # EDEN 推理封装
      scheduler/                # greedy_refine 等
      modules/                  # attention（含 xformers fallback）
  third_party/
    wan/wan/                    # WAN 源码（保持 import wan）
    raft/                       # RAFT 相关代码
    vbench/                     # VBench（可选）
  docs/legacy/                  # 历史说明文档（不影响主流程）
```

---

## 2. 环境安装

### 2.1 Python & PyTorch（建议）

* Python 3.10
* CUDA 对应的 PyTorch（你环境是 torch 2.4.x + cu124 也可以）

### 2.2 安装依赖

```bash
pip install -r requirements.txt
```

> 注意：本项目写视频使用 PyAV。若你的环境 PyAV 版本较新（例如 14.x），也能运行，因为我们已绕过 torchvision 写视频接口。

---

## 3. 模型与权重准备

你需要准备以下权重（建议不要提交到 git，`.gitignore` 已忽略）：

* WAN checkpoint（示例路径：`/data/chenjiayu/hengyi_zhang/pretrained_models/Wan2.1-T2V-14B`，以你实际下载的为准）
* RAFT 权重（示例路径：`/data/chenjiayu/hengyi_zhang/pretrained_models/raft/raft-things.pth`）
* EDEN 权重（由 `--eden_config` 内部配置指定）

---

## 4. 快速开始：完整 pipeline（WAN → 8fps 采样 → EDEN → 24fps）

示例命令（生成中文 prompt，并把 WAN 输出按 8fps 均匀采样，然后插到 24fps）：

```bash
python scripts/run_full_pipeline.py \
  --wan_ckpt_dir /data/chenjiayu/hengyi_zhang/pretrained_models/Wan2.1-T2V-14B \
  --prompt "一只白猫和一只黑猫在打架" \
  --eden_config configs/eval_eden.yaml \
  --raft_ckpt /data/chenjiayu/hengyi_zhang/pretrained_models/raft/raft-things.pth \
  --wan_out_fps 8 \
  --wan_frame_sample uniform \
  --keyframe_mode all \
  --target_fps 24 \
  --output_path interpolation_outputs/final.mp4 \
  --save_wan_video interpolation_outputs/wan_8fps.mp4
```

参数解释：

* `--wan_out_fps 8`：把 WAN 输出重采样为 8fps 的关键帧序列（保持时长不变）
* `--wan_frame_sample uniform`：均匀采样（也支持 random/stratified_random）
* `--keyframe_mode all`：**重要**：因为关键帧已经在 WAN 阶段产生，所以插帧阶段不再二次采样
* `--target_fps 24`：最终输出帧率

---

## 5. 从已有视频开始插帧（跳过 WAN）

```bash
python scripts/run_full_pipeline.py \
  --input_video path/to/input.mp4 \
  --eden_config configs/eval_eden.yaml \
  --raft_ckpt /data/chenjiayu/hengyi_zhang/pretrained_models/raft/raft-things.pth \
  --keyframe_mode all \
  --target_fps 24 \
  --output_path interpolation_outputs/out_24fps.mp4
```

---

## 6. 保存中间视频（强烈推荐用于调试）

* `--save_wan_video <path>`：保存进入插帧前的“采样后视频”（例如 wan_out_fps=8 的结果）
* `--save_sampled_video <path>`：同上（建议逐步统一只保留该参数）
* `--save_keyframes_video <path>`：保存插帧阶段选出的关键帧预览视频（当 keyframe_mode=uniform/random 时有用）

---

## 7. xFormers / 注意力加速说明

如果你的环境 xformers 没有 CUDA 扩展，或输入 dtype 不支持（常见于 float32 推理），项目会自动 fallback 到 PyTorch 的 `scaled_dot_product_attention`，保证流程可跑通（但速度会慢一些）。

---

## 8. 如何新增生成器（例如未来接入 CogVideo）

本项目对生成器做了插件化抽象：

* 接口：`vdit.generators.base.VideoGenerator`
* 注册：`@register_generator("name")`
* 创建：`create_generator(name, **kwargs)`

新增 CogVideo 时，只需要：

1. 新增文件 `src/vdit/generators/cogvideo.py`
2. 实现 `generate(prompt) -> (frames[T,3,H,W], fps)`
3. 注册 `@register_generator("cogvideo")`
4. 在脚本中增加对应参数并传入 `generator_name="cogvideo"`

插帧 pipeline 与后处理无需改动。

