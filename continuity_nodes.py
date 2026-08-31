"""Director-owned persistence and delivery nodes for H3 AV continuation."""

from __future__ import annotations

import logging
import os
from pathlib import Path

import comfy.nested_tensor
import folder_paths

try:
    from safetensors.torch import load_file as _load_safetensors
    from safetensors.torch import save_file as _save_safetensors
except ImportError:  # pragma: no cover - ComfyUI normally provides safetensors
    _load_safetensors = None
    _save_safetensors = None

from .vendor.h3_masked import (
    MiniMaxH3ExistingVideoMaskedContext,
    MiniMaxH3GeneratedAVMaskedContext,
    MiniMaxH3MaskedAVBridge,
)
from .vendor.h3_motion import H3DirectorMotionCondition


_LOG = logging.getLogger("h3_director.continuity")


def _av_streams(latent):
    samples = latent.get("samples")
    if hasattr(samples, "unbind"):
        streams = list(samples.unbind())
    elif isinstance(samples, (tuple, list)):
        streams = list(samples)
    else:
        raise ValueError("H3 导演台：需要 MiniMax H3 联合音视频 latent。")
    if len(streams) < 2:
        raise ValueError("H3 导演台：latent 缺少音频流。")
    return streams[0], streams[1]


def _resolve_slot(latent_path: str, clip_index: int) -> Path:
    raw = (latent_path or "").strip().strip('"').strip("'")
    root = Path(folder_paths.get_output_directory()).resolve()
    candidate = Path(raw)
    candidate = candidate.resolve() if candidate.is_absolute() else (root / candidate).resolve()
    if candidate != root and root not in candidate.parents:
        raise ValueError("H3 导演台：连续性 latent 路径必须位于 ComfyUI 输出目录内。")
    if candidate.is_file():
        return candidate
    if not candidate.is_dir():
        raise FileNotFoundError(f"H3 导演台：连续性目录不存在：{candidate}")

    index = int(clip_index)
    if index > 0:
        matches = sorted(candidate.glob(f"*_{index:05d}.safetensors"))
    else:
        matches = sorted(candidate.glob("*.safetensors"), key=lambda item: item.stat().st_mtime_ns)
    if not matches:
        raise FileNotFoundError(f"H3 导演台：找不到镜头 {index} 的连续性 latent：{candidate}")
    return max(matches, key=lambda item: item.stat().st_mtime_ns)


class H3DirectorSaveAVLatent:
    @classmethod
    def INPUT_TYPES(cls):
        return {
            "required": {
                "latent": ("LATENT",),
                "filename_prefix": ("STRING", {"default": "director_console/context/clip"}),
                "clip_index": ("INT", {"default": 1, "min": 1, "max": 9999}),
            }
        }

    RETURN_TYPES = ("STRING",)
    RETURN_NAMES = ("latent_path",)
    FUNCTION = "save"
    OUTPUT_NODE = True
    CATEGORY = "MiniMax H3/Director Console/Internal"

    def save(self, latent, filename_prefix, clip_index=1):
        if _save_safetensors is None:
            raise RuntimeError("H3 导演台：safetensors 不可用，无法保存连续性 latent。")
        video, audio = _av_streams(latent)
        folder, filename, _, _, _ = folder_paths.get_save_image_path(
            filename_prefix, folder_paths.get_output_directory()
        )
        path = Path(folder) / f"{filename}_{int(clip_index):05d}.safetensors"
        _save_safetensors(
            {"video": video.detach().cpu().contiguous(), "audio": audio.detach().cpu().contiguous()},
            str(path),
            metadata={"format": "h3_director_av_v1"},
        )
        _LOG.info("saved clip %d AV latent to %s", int(clip_index), path)
        return (str(path),)


class H3DirectorLatentTailFrame:
    """Decode the exact final frame of a saved H3 video latent."""

    @classmethod
    def INPUT_TYPES(cls):
        return {
            "required": {
                "latent": ("LATENT",),
                "vae": ("VAE",),
            }
        }

    RETURN_TYPES = ("IMAGE",)
    RETURN_NAMES = ("last_frame",)
    FUNCTION = "decode"
    CATEGORY = "MiniMax H3/Director Console/Internal"
    DESCRIPTION = "Decode the accepted previous clip latent and return only its final frame."

    def decode(self, latent, vae):
        video, _audio = _av_streams(latent)
        images = vae.decode(video)
        if len(images.shape) == 5:
            images = images.reshape(-1, images.shape[-3], images.shape[-2], images.shape[-1])
        if int(images.shape[0]) < 1:
            raise ValueError("H3 导演台：上一镜头 latent 未解码出可用画面。")
        return (images[-1:].contiguous(),)


class H3DirectorLoadAVLatent:
    @classmethod
    def INPUT_TYPES(cls):
        return {
            "required": {
                "latent_path": ("STRING", {"default": "director_console/context"}),
                "clip_index": ("INT", {"default": 1, "min": 1, "max": 9999}),
            }
        }

    RETURN_TYPES = ("LATENT",)
    FUNCTION = "load"
    CATEGORY = "MiniMax H3/Director Console/Internal"

    @classmethod
    def IS_CHANGED(cls, latent_path, clip_index=1):
        try:
            path = _resolve_slot(latent_path, clip_index)
            return f"{path}:{path.stat().st_mtime_ns}"
        except Exception:
            return float("NaN")

    def load(self, latent_path, clip_index=1):
        if _load_safetensors is None:
            raise RuntimeError("H3 导演台：safetensors 不可用，无法读取连续性 latent。")
        path = _resolve_slot(latent_path, clip_index)
        data = _load_safetensors(str(path))
        if "video" not in data or "audio" not in data:
            raise ValueError(f"H3 导演台：{path} 不是联合音视频 latent 文件。")
        return ({"samples": comfy.nested_tensor.NestedTensor((data["video"], data["audio"]))},)


class H3DirectorTrimAV:
    @classmethod
    def INPUT_TYPES(cls):
        return {
            "required": {
                "images": ("IMAGE",),
                "trim_frames": ("INT", {"default": 0, "min": 0, "max": 4096}),
                "audio": ("AUDIO",),
                "fps": ("FLOAT", {"default": 24.0, "min": 1.0, "max": 240.0}),
            },
            "optional": {"match_tail": ("BOOLEAN", {"default": True})},
        }

    RETURN_TYPES = ("IMAGE", "AUDIO")
    RETURN_NAMES = ("images", "audio")
    FUNCTION = "trim"
    CATEGORY = "MiniMax H3/Director Console/Internal"

    def trim(self, images, trim_frames, audio, fps=24.0, match_tail=True):
        count = max(0, int(trim_frames))
        total = int(images.shape[0])
        if count >= total:
            raise ValueError(f"H3 导演台：不能从 {total} 帧中裁掉 {count} 帧。")
        result_images = images[count:] if count else images
        waveform = audio["waveform"]
        sample_rate = int(audio["sample_rate"])
        cut = int(round(count / float(fps) * sample_rate))
        if cut >= int(waveform.shape[-1]):
            raise ValueError("H3 导演台：连续性音频裁切后将为空。")
        waveform = waveform[..., cut:]
        if match_tail:
            wanted = int(round(int(result_images.shape[0]) / float(fps) * sample_rate))
            if int(waveform.shape[-1]) > wanted:
                waveform = waveform[..., :wanted]
        return (result_images, {"waveform": waveform, "sample_rate": sample_rate})


NODE_CLASS_MAPPINGS = {
    "H3DirectorMotionCondition": H3DirectorMotionCondition,
    "H3DirectorGeneratedAVMaskedContext": MiniMaxH3GeneratedAVMaskedContext,
    "H3DirectorExistingVideoMaskedContext": MiniMaxH3ExistingVideoMaskedContext,
    "H3DirectorMaskedAVBridge": MiniMaxH3MaskedAVBridge,
    "H3DirectorLatentTailFrame": H3DirectorLatentTailFrame,
    "H3DirectorSaveAVLatent": H3DirectorSaveAVLatent,
    "H3DirectorLoadAVLatent": H3DirectorLoadAVLatent,
    "H3DirectorTrimAV": H3DirectorTrimAV,
}

NODE_DISPLAY_NAME_MAPPINGS = {
    "H3DirectorMotionCondition": "H3 导演台 · 内置运动连续性",
    "H3DirectorGeneratedAVMaskedContext": "H3 导演台 · 内置 Latent 连续性",
    "H3DirectorExistingVideoMaskedContext": "H3 导演台 · 已有视频续写",
    "H3DirectorMaskedAVBridge": "H3 导演台 · 双端过渡修复",
    "H3DirectorLatentTailFrame": "H3 导演台 · 上一镜头构图锚点",
    "H3DirectorSaveAVLatent": "H3 导演台 · 保存连续性状态",
    "H3DirectorLoadAVLatent": "H3 导演台 · 读取连续性状态",
    "H3DirectorTrimAV": "H3 导演台 · 裁切重复上下文",
}
