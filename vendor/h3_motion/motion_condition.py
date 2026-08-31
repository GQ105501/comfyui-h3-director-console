"""Director-owned latent motion conditioning for MiniMax H3 clip chains.

Adapted from NikoDemon80/ComfyUI-H3-Motion-Context (GPL-3.0). This version
accepts only generated H3 AV latents and therefore never decodes, re-encodes,
or depends on an external custom-node package.
"""

from __future__ import annotations

import logging

import comfy.ldm.common_dit
import node_helpers

from .patch_layout import MC_KEY, apply_patch as _apply_layout_patch
from .patch_payload import apply_patch as _apply_payload_patch


FRAME_PER_TOKEN = (1, 4, 4, 4, 4)
VIDEO_PATCH_SIZE = (1, 2, 2)
_LOG = logging.getLogger("h3_director.motion")


def _streams(latent):
    samples = latent.get("samples")
    if hasattr(samples, "unbind"):
        parts = list(samples.unbind())
    elif isinstance(samples, (tuple, list)):
        parts = list(samples)
    else:
        raise ValueError("H3 导演台：需要 MiniMax H3 联合音视频 latent。")
    if len(parts) < 2:
        raise ValueError("H3 导演台：连续性 latent 缺少音频流。")
    video = parts[0]
    if video.ndim == 4:
        video = video.unsqueeze(0)
    if video.ndim != 5:
        raise ValueError(f"H3 导演台：视频 latent 形状无效：{tuple(video.shape)}")
    return video, parts[1]


def _pixel_frames(latent_steps: int) -> int:
    return sum(FRAME_PER_TOKEN[index % 5] for index in range(int(latent_steps)))


def _steps_for_frames(frame_count: int) -> int | None:
    steps = covered = 0
    while covered < int(frame_count):
        covered += FRAME_PER_TOKEN[steps % 5]
        steps += 1
    return steps if covered == int(frame_count) else None


def _step_offsets(latent_steps: int) -> list[int]:
    offsets = []
    covered = 0
    for index in range(int(latent_steps)):
        offsets.append(covered)
        covered += FRAME_PER_TOKEN[index % 5]
    return offsets


def _ensure_patches():
    if not _apply_layout_patch():
        raise RuntimeError("H3 导演台：无法启用中间帧运动锚点。请检查启动日志。")
    if not _apply_payload_patch():
        raise RuntimeError("H3 导演台：无法合并运动锚点与多参考条件。请检查启动日志。")


class H3DirectorMotionCondition:
    @classmethod
    def INPUT_TYPES(cls):
        return {
            "required": {
                "conditioning": ("CONDITIONING",),
                "latent": ("LATENT", {"tooltip": "当前镜头的目标 H3 AV latent。"}),
                "source_latent": (
                    "LATENT",
                    {"tooltip": "上一镜头保存的 H3 AV latent；直接取尾部运动 token。"},
                ),
                "context_length": (
                    "INT",
                    {
                        "default": 39,
                        "min": 5,
                        "max": 192,
                        "tooltip": "推荐 39 帧。必须落在 H3 的 17k+5 时间网格。",
                    },
                ),
            }
        }

    RETURN_TYPES = ("CONDITIONING", "INT")
    RETURN_NAMES = ("conditioning", "trim_frames")
    FUNCTION = "apply"
    CATEGORY = "MiniMax H3/Director Console/Internal"
    DESCRIPTION = (
        "把上一段尾部连续视频 latent 作为当前段开头的多帧 H3 锚点；"
        "与 Ref2VA 参考和 Masked Latent 同时工作。"
    )

    def apply(self, conditioning, latent, source_latent, context_length=39):
        _ensure_patches()
        target_video, _target_audio = _streams(latent)
        source_video, _source_audio = _streams(source_latent)

        if int(target_video.shape[0]) != 1 or int(source_video.shape[0]) != 1:
            raise ValueError("H3 导演台：运动连续性目前只支持 batch size 1。")
        if int(target_video.shape[1]) != int(source_video.shape[1]):
            raise ValueError("H3 导演台：前后镜头的视频 latent 通道不一致。")
        if tuple(target_video.shape[-2:]) != tuple(source_video.shape[-2:]):
            raise ValueError(
                "H3 导演台：前后镜头分辨率不一致；latent 无法安全缩放后续写。"
            )

        frame_count = _pixel_frames(int(target_video.shape[2]))
        available = _pixel_frames(int(source_video.shape[2]))
        requested = min(int(context_length), available)
        steps = _steps_for_frames(requested)
        if steps is None:
            raise ValueError(
                f"H3 导演台：{requested} 帧不在 H3 17k+5 连续性网格上。"
            )
        if requested >= frame_count:
            raise ValueError("H3 导演台：运动上下文必须短于当前目标镜头。")

        start = int(source_video.shape[2]) - steps
        if start < 0 or start % 5 != 0:
            raise ValueError(
                "H3 导演台：上一镜头尾部的 latent token 相位不匹配，拒绝错帧续写。"
            )

        keyframes = []
        for index, frame_index in enumerate(_step_offsets(steps)):
            block = source_video[:1, :, start + index : start + index + 1].clone()
            block = comfy.ldm.common_dit.pad_to_patch_size(block, VIDEO_PATCH_SIZE)
            keyframes.append(
                {
                    "resolved_frame_index": 0,
                    MC_KEY: frame_index,
                    "latent": block,
                }
            )

        values = {
            "minimax_keyframes": keyframes,
            "minimax_frame_count": frame_count,
        }
        out = node_helpers.conditioning_set_values(conditioning, values)
        _LOG.info(
            "H3 Director: injected %d motion tokens covering %d frames into a %d-frame target",
            len(keyframes),
            requested,
            frame_count,
        )
        return (out, requested)
