"""Build compact MiniMax H3 API prompts for the Director Console.

Generation modality and acceleration are intentionally independent. The mode
controls which encoder and assets are wired; the acceleration switch controls
whether a compatible LoRA is attached and which sampling settings are used.
"""

from __future__ import annotations

import re
import math
from copy import deepcopy
from dataclasses import dataclass
from pathlib import PurePosixPath
from typing import Any


DEFAULT_MODELS = {
    "ref_model": "minimax_h3_ref2va_pruned_int8_convrot.safetensors",
    "fl_model": "minimax_h3_fl2va_pruned_int8_convrot.safetensors",
    "clip": "qwen3vl_32b_minimax_h3_nvfp4_awq.safetensors",
    "video_vae": "minimax_h3_video_vae_fp16.safetensors",
    "audio_vae": "minimax_h3_audio_vae_fp32.safetensors",
    "ref_lora": "minimax_h3_ref2v_turbo_4step_v0.1_comfyui_bf16.safetensors",
    "fl_lora": "minimax_h3_fl2v_turbo_4step_v1.1_768p_comfyui_bf16.safetensors",
    "upscaler": "minimax_h3_latent_upscaler_3d_fp32.pth",
}

MODES = {
    "t2v": {
        "label": "文生视频",
        "description": "只使用提示词，不接入参考素材",
        "model": "fl_model",
        "compatible_loras": ["fl2v_v1_1"],
    },
    "i2v": {
        "label": "图生视频",
        "description": "提示词 + 首帧；可选尾帧，最多 2 张图片",
        "model": "fl_model",
        "compatible_loras": ["fl2v_v1_1"],
    },
    "ref2v": {
        "label": "多参考生视频",
        "description": "提示词 + 图片、视频、音频，总计最多 12 个参考文件",
        "model": "ref_model",
        "compatible_loras": ["ref2v_v0_1", "fl2v_v1_1"],
    },
}

ACCELERATION_LORAS = {
    "fl2v_v1_1": {
        "label": "MiniMax H3 FL2V Turbo 4步 v1.1",
        "model": "fl_lora",
        "compatible_modes": ["t2v", "i2v", "ref2v"],
        "experimental_modes": ["ref2v"],
        "recommended_steps": 4,
        "sampler": "euler",
        "scheduler": "simple",
        "shift_video": 6.0,
        "shift_audio": 3.0,
    },
    "ref2v_v0_1": {
        "label": "MiniMax H3 Ref2V Turbo 4步 v0.1",
        "model": "ref_lora",
        "compatible_modes": ["ref2v"],
        "experimental_modes": [],
        "recommended_steps": 4,
        "sampler": "euler",
        "scheduler": "simple",
        "shift_video": 6.0,
        "shift_audio": 3.0,
    },
}

ASPECT_RATIOS = {
    "16_9": {"label": "横屏 16:9", "width": 16, "height": 9},
    "9_16": {"label": "竖屏 9:16", "width": 9, "height": 16},
    "1_1": {"label": "方形 1:1", "width": 1, "height": 1},
    "4_3": {"label": "横屏 4:3", "width": 4, "height": 3},
    "3_4": {"label": "竖屏 3:4", "width": 3, "height": 4},
    "21_9": {"label": "电影宽屏 21:9", "width": 21, "height": 9},
}

MEGAPIXELS = {
    "0_2": {"label": "0.2 MP · 极速草稿", "value": 0.2},
    "0_3": {"label": "0.3 MP", "value": 0.3},
    "0_4": {"label": "0.4 MP", "value": 0.4},
    "0_5": {"label": "0.5 MP", "value": 0.5},
    "0_6": {"label": "0.6 MP · 12GB 推荐", "value": 0.6},
    "0_7": {"label": "0.7 MP", "value": 0.7},
    "0_8": {"label": "0.8 MP", "value": 0.8},
    "0_9": {"label": "0.9 MP", "value": 0.9},
    "0_98": {"label": "0.98 MP · 官方 768p 档", "value": 0.98},
}

OUTPUT_QUALITIES = {
    "720p": {"label": "720P · 高清", "short_edge": 720},
    "1080p": {"label": "1080P · 全高清", "short_edge": 1080},
    "1440p": {"label": "1440P / 2K · 超清", "short_edge": 1440},
    "2160p": {"label": "2160P / 4K · 超高清", "short_edge": 2160},
}

SEQUENCE_MODES = {
    "continuous": {
        "label": "连续长镜头",
        "description": "把所有片段视为同一次拍摄；锁定机位、景别、人物比例和运动方向。",
    },
    "shots": {
        "label": "独立分镜",
        "description": "每段独立生成；允许切换景别、机位和镜头运动，后期按正常剪辑连接。",
    },
}

DEFAULT_SETTINGS = {
    "mode": "ref2v",
    "aspect_ratio": "16_9",
    "megapixels": "0_6",
    "duration_seconds": 10.0,
    "fps": 24,
    "acceleration_enabled": False,
    "acceleration_lora": "ref2v_v0_1",
    "steps": 20,
    "sampler": "res_multistep",
    "scheduler": "beta",
    "lora_strength": 1.0,
    "seed": 2026082601,
    "seed_mode": "increment",
    "sequence_mode": "continuous",
    "continuity": True,
    "continuity_strategy": "masked_latent",
    "context_frames": 39,
    "video_feather_tokens": 0,
    "audio_feather_ticks": 8,
    "continuity_prompt_lock": True,
    "second_pass": False,
    "output_quality": "720p",
    "refine_steps": 2,
    "models": DEFAULT_MODELS,
}

DEFAULT_INFERENCE = {
    "steps": 20,
    "sampler": "res_multistep",
    "scheduler": "beta",
    "shift_video": 12.0,
    "shift_audio": 3.0,
}

MAX_ASSETS = {"images": 9, "videos": 3, "audios": 3}
SAFE_ID = re.compile(r"[^a-zA-Z0-9_-]+")


class DirectorValidationError(ValueError):
    """Raised when a project cannot be translated into a safe prompt."""


@dataclass
class BuildResult:
    prompt: dict[str, dict[str, Any]]
    output_prefix: str
    shot_index: int
    mode: str
    warnings: list[str]


class Graph:
    def __init__(self) -> None:
        self.nodes: dict[str, dict[str, Any]] = {}
        self._next = 1

    def add(self, class_type: str, inputs: dict[str, Any], title: str) -> tuple[str, int]:
        node_id = str(self._next)
        self._next += 1
        self.nodes[node_id] = {"class_type": class_type, "inputs": inputs, "_meta": {"title": title}}
        return node_id, 0


def slug(value: str, fallback: str) -> str:
    cleaned = SAFE_ID.sub("-", (value or "").strip()).strip("-_")
    return (cleaned or fallback)[:80]


def align_frames(value: int) -> int:
    value = max(5, min(3600, int(value)))
    while value % 17 != 5:
        value += 1
    return value


H3_MASKED_CONTEXT_WINDOWS = (192, 141, 90, 39)
H3_GUIDE_CONTEXT_WINDOWS = (56, 39, 22, 5)


def effective_context_frames(value: int, strategy: str = "masked_latent") -> int:
    """Resolve a requested overlap to the selected H3 AV-safe grid."""
    value = max(0, int(value))
    if value == 0:
        return 0
    if strategy == "masked_latent":
        return next((window for window in H3_MASKED_CONTEXT_WINDOWS if window <= value), 39)
    return next((window for window in H3_GUIDE_CONTEXT_WINDOWS if window <= value), 5)


def prompt_with_continuity_contract(
    prompt: str,
    shot_index: int,
    settings: dict[str, Any],
    anchor_picture: bool = False,
) -> str:
    """Inject a camera/layout hand-off contract into continuation prompts."""
    sequence_mode = settings.get("sequence_mode")
    continuous = sequence_mode == "continuous" or (
        sequence_mode is None and settings.get("continuity")
    )
    if shot_index == 0 or not continuous or not settings.get("continuity_prompt_lock", True):
        return prompt
    if anchor_picture:
        prompt = re.sub(
            r"<Picture\s+(\d+)>",
            lambda match: f"<Picture {int(match.group(1)) + 1}>",
            prompt,
        )
        anchor_definition = (
            "<Picture 1> is the exact final delivered frame of the accepted previous segment. "
            "It is the sole authority for camera geometry, shot size, subject scale, pose, screen "
            "coordinates, background layout, lighting, and weapon state. Other pictures may preserve "
            "identity or design details but must never replace Picture 1's composition.\n"
        )
        marker = "subject_definitions:\n"
        prompt = (
            prompt.replace(marker, marker + anchor_definition, 1)
            if marker in prompt
            else anchor_definition + prompt
        )
    contract = (
        "This segment is not a new shot; it is the next uninterrupted part of the same master take. "
        "The inherited latent tail is physical truth. Continue its exact camera translation, rotation, "
        "zoom velocity, subject motion, pose velocity, cloth inertia, wind, lighting, and audio phase. "
        "If the inherited camera is nearly static, keep it nearly static. Preserve camera height, focal "
        "length, camera-to-subject distance, subject scale, screen coordinates, horizon, vanishing point, "
        "and background anchors throughout every delivered frame. Do not cut, reset, dissolve, crossfade, "
        "or recompose the inherited shot. Any instruction in the stored shot prompt that requests or "
        "implies a new shot size, close-up, wide shot, zoom, dolly, pan, tilt, "
        "orbit, crane, reframe, angle, lens, cut, dissolve, crossfade, or camera reset is overridden and "
        "must be ignored. Never morph, double expose, ghost, or interpolate between two compositions. "
        "During the first twelve delivered frames, ease the new action out of the inherited motion without "
        "changing composition or restarting the pose. The protected continuity prefix is removed before "
        "delivery. Treat every negative or limiting instruction in the stored shot prompt as a hard "
        "frame-by-frame constraint. Do not repeat or reset an action already completed in the inherited "
        "prefix. Do not perform an action reserved for a later shot or segment. "
    )
    for marker in ("detailed_description:\n", "integrated_multimodal_description:\n"):
        if marker in prompt:
            return prompt.replace(marker, marker + contract, 1)
    return contract + prompt


def duration_to_frames(seconds: float, fps: int = 24) -> int:
    """Convert a user-facing duration to the H3-required 17k+5 frame grid."""
    seconds = max(1.0, min(15.0, float(seconds)))
    return align_frames(round(seconds * fps))


def align_dimension(value: int) -> int:
    return max(32, min(4096, ((int(value) + 16) // 32) * 32))


def dimensions_for_megapixels(aspect_ratio: str, megapixels: str) -> tuple[int, int]:
    """Return a 32-aligned canvas using Comfy's MiP-style megapixel scale."""
    aspect = ASPECT_RATIOS[aspect_ratio]
    ratio = aspect["width"] / aspect["height"]
    pixels = MEGAPIXELS[megapixels]["value"] * 1024 * 1024
    height = math.sqrt(pixels / ratio)
    width = height * ratio
    return align_dimension(width), align_dimension(height)


def dimensions_for_quality(aspect_ratio: str, output_quality: str) -> tuple[int, int]:
    """Apply a familiar P-tier to the short edge while preserving aspect ratio."""
    aspect = ASPECT_RATIOS[aspect_ratio]
    ratio = aspect["width"] / aspect["height"]
    short_edge = OUTPUT_QUALITIES[output_quality]["short_edge"]
    if ratio >= 1:
        width, height = short_edge * ratio, short_edge
    else:
        width, height = short_edge, short_edge / ratio
    if max(width, height) > 4096:
        scale = 4096 / max(width, height)
        width *= scale
        height *= scale
    return align_dimension(width), align_dimension(height)


def safe_asset_path(value: str) -> str:
    path = PurePosixPath((value or "").replace("\\", "/"))
    if path.is_absolute() or ".." in path.parts or not path.name:
        raise DirectorValidationError(f"不安全的素材路径：{value}")
    return path.as_posix()


def _closest_aspect_ratio(width: int, height: int) -> str:
    ratio = width / height
    return min(
        ASPECT_RATIOS,
        key=lambda name: abs(ratio - ASPECT_RATIOS[name]["width"] / ASPECT_RATIOS[name]["height"]),
    )


def _closest_megapixels(width: int, height: int) -> str:
    value = width * height / (1024 * 1024)
    return min(MEGAPIXELS, key=lambda name: abs(value - MEGAPIXELS[name]["value"]))


def _closest_output_quality(aspect_ratio: str, width: int, height: int) -> str:
    area = width * height
    return min(
        OUTPUT_QUALITIES,
        key=lambda name: abs(area - math.prod(dimensions_for_quality(aspect_ratio, name))),
    )


def normalize_project(project: dict[str, Any]) -> dict[str, Any]:
    if not isinstance(project, dict):
        raise DirectorValidationError("项目数据必须是 JSON 对象")
    result = deepcopy(project)
    result["version"] = 4
    result["id"] = slug(str(result.get("id", "")), "director-project")
    result["name"] = str(result.get("name") or "未命名导演台项目")[:120]

    incoming = result.get("settings") or {}
    if not isinstance(incoming, dict):
        incoming = {}
    legacy_mode = str(incoming.get("mode") or "")
    settings = deepcopy(DEFAULT_SETTINGS)
    settings.update(incoming)
    settings["models"] = {**DEFAULT_MODELS, **(settings.get("models") or {})}

    legacy_modes = {"ref2va_standard": "ref2v", "ref2v_turbo": "ref2v", "fl2v_v1_1": "i2v"}
    settings["mode"] = legacy_modes.get(legacy_mode, settings.get("mode"))
    if settings["mode"] not in MODES:
        settings["mode"] = DEFAULT_SETTINGS["mode"]

    if "acceleration_enabled" not in incoming:
        settings["acceleration_enabled"] = legacy_mode in {"ref2v_turbo", "fl2v_v1_1"}
    if "acceleration_lora" not in incoming:
        settings["acceleration_lora"] = "fl2v_v1_1" if legacy_mode == "fl2v_v1_1" else "ref2v_v0_1"
    settings["acceleration_enabled"] = bool(settings["acceleration_enabled"])
    if settings["acceleration_lora"] not in ACCELERATION_LORAS:
        settings["acceleration_lora"] = DEFAULT_SETTINGS["acceleration_lora"]

    legacy_resolutions = {
        "wide_0_6": ("16_9", "0_6"),
        "wide_1_0": ("16_9", "0_98"),
        "portrait_0_6": ("9_16", "0_6"),
        "portrait_1_0": ("9_16", "0_98"),
        "square_0_6": ("1_1", "0_6"),
        "square_1_0": ("1_1", "0_98"),
    }
    legacy_resolution = legacy_resolutions.get(str(incoming.get("resolution") or ""))
    try:
        legacy_width = align_dimension(incoming.get("width", 1024))
        legacy_height = align_dimension(incoming.get("height", 576))
    except (TypeError, ValueError):
        legacy_width, legacy_height = 1024, 576
    settings["aspect_ratio"] = str(settings.get("aspect_ratio") or "")
    if "aspect_ratio" not in incoming or settings["aspect_ratio"] not in ASPECT_RATIOS:
        settings["aspect_ratio"] = legacy_resolution[0] if legacy_resolution else _closest_aspect_ratio(legacy_width, legacy_height)
    settings["megapixels"] = str(settings.get("megapixels") or "")
    if "megapixels" not in incoming or settings["megapixels"] not in MEGAPIXELS:
        settings["megapixels"] = legacy_resolution[1] if legacy_resolution else _closest_megapixels(legacy_width, legacy_height)
    settings["width"], settings["height"] = dimensions_for_megapixels(settings["aspect_ratio"], settings["megapixels"])

    settings["output_quality"] = str(settings.get("output_quality") or "")
    if "output_quality" not in incoming or settings["output_quality"] not in OUTPUT_QUALITIES:
        try:
            target_width = align_dimension(incoming.get("target_width", 1280))
            target_height = align_dimension(incoming.get("target_height", 720))
        except (TypeError, ValueError):
            target_width, target_height = 1280, 736
        settings["output_quality"] = _closest_output_quality(settings["aspect_ratio"], target_width, target_height)
    settings["target_width"], settings["target_height"] = dimensions_for_quality(
        settings["aspect_ratio"], settings["output_quality"]
    )
    if settings.get("second_pass") and settings["target_width"] * settings["target_height"] <= settings["width"] * settings["height"]:
        settings["output_quality"] = next(
            (
                name
                for name in OUTPUT_QUALITIES
                if math.prod(dimensions_for_quality(settings["aspect_ratio"], name)) > settings["width"] * settings["height"]
            ),
            "2160p",
        )
        settings["target_width"], settings["target_height"] = dimensions_for_quality(
            settings["aspect_ratio"], settings["output_quality"]
        )
    settings.pop("resolution", None)
    settings.pop("target_resolution", None)

    settings["fps"] = max(1, min(60, int(settings.get("fps", 24))))
    if "duration_seconds" not in incoming and "frames" in incoming:
        settings["duration_seconds"] = float(incoming["frames"]) / settings["fps"]
    settings["duration_seconds"] = max(1.0, min(15.0, float(settings["duration_seconds"])))
    settings["frames"] = duration_to_frames(settings["duration_seconds"], settings["fps"])

    if settings["acceleration_enabled"]:
        settings["steps"] = max(1, min(50, int(settings["steps"])))
        settings["sampler"] = str(settings["sampler"])
        settings["scheduler"] = str(settings["scheduler"])
    else:
        settings.update({key: DEFAULT_INFERENCE[key] for key in ("steps", "sampler", "scheduler")})
    settings["lora_strength"] = max(0.0, min(2.0, float(settings["lora_strength"])))
    settings["refine_steps"] = max(1, min(settings["steps"] - 1 if settings["steps"] > 1 else 1, int(settings["refine_steps"])))
    settings["sequence_mode"] = str(settings.get("sequence_mode") or "")
    if "sequence_mode" not in incoming:
        settings["sequence_mode"] = "continuous" if bool(settings.get("continuity", True)) else "shots"
    elif settings["sequence_mode"] not in SEQUENCE_MODES:
        settings["sequence_mode"] = "continuous" if bool(settings.get("continuity", True)) else "shots"
    settings["continuity_strategy"] = str(settings.get("continuity_strategy") or "masked_latent")
    if settings["continuity_strategy"] not in {"masked_latent", "motion_context"}:
        settings["continuity_strategy"] = "masked_latent"
    settings["context_frames"] = max(0, min(192, int(settings["context_frames"])))
    settings["video_feather_tokens"] = max(0, min(16, int(settings.get("video_feather_tokens", 0))))
    settings["audio_feather_ticks"] = max(0, min(64, int(settings.get("audio_feather_ticks", 8))))
    settings["continuity_prompt_lock"] = bool(settings.get("continuity_prompt_lock", True))
    if settings["sequence_mode"] == "continuous":
        settings["continuity"] = True
        settings["continuity_strategy"] = "masked_latent"
        settings["context_frames"] = 39
        settings["continuity_prompt_lock"] = True
        settings["video_feather_tokens"] = 0
        settings["audio_feather_ticks"] = 8
    else:
        settings["continuity"] = False
    settings["second_pass"] = bool(settings["second_pass"])
    result["settings"] = settings

    shots = result.get("shots") or []
    if not isinstance(shots, list) or not shots:
        raise DirectorValidationError("导演台至少需要一个镜头")
    normalized = []
    seen = set()
    for index, original in enumerate(shots):
        shot = deepcopy(original if isinstance(original, dict) else {})
        shot_id = slug(str(shot.get("id", "")), f"shot-{index + 1:03d}")
        if shot_id in seen:
            shot_id = f"{shot_id}-{index + 1}"
        seen.add(shot_id)
        shot["id"] = shot_id
        shot["title"] = str(shot.get("title") or f"镜头 {index + 1:03d}")[:120]
        shot["prompt"] = str(shot.get("prompt") or "")[:100_000]
        shot["enabled"] = bool(shot.get("enabled", True))
        assets = shot.get("assets") or {}
        clean_assets: dict[str, list[dict[str, str]]] = {}
        for kind, limit in MAX_ASSETS.items():
            items = assets.get(kind) or []
            if not isinstance(items, list):
                items = []
            clean_assets[kind] = [
                {"path": safe_asset_path(str(item.get("path", ""))), "name": str(item.get("name") or PurePosixPath(str(item.get("path", ""))).name)[:180]}
                for item in items[:limit]
                if isinstance(item, dict) and item.get("path")
            ]
        if sum(len(value) for value in clean_assets.values()) > 12:
            raise DirectorValidationError(f"{shot['title']} 的参考素材超过 H3 的 12 文件上限")
        shot["assets"] = clean_assets
        normalized.append(shot)
    result["shots"] = normalized
    return result


def _seed(settings: dict[str, Any], shot: dict[str, Any], index: int) -> int:
    if shot.get("seed") is not None:
        return int(shot["seed"]) % (2**64)
    base = int(settings["seed"]) % (2**64)
    return (base + index) % (2**64) if settings["seed_mode"] == "increment" else base


def _load_assets(graph: Graph, shot: dict[str, Any], mode_name: str) -> dict[str, list[tuple[str, int]]]:
    loaded: dict[str, list[tuple[str, int]]] = {"images": [], "videos": [], "video_audios": [], "audios": []}
    image_items = shot["assets"]["images"][:2] if mode_name == "i2v" else shot["assets"]["images"]
    if mode_name in {"i2v", "ref2v"}:
        for index, item in enumerate(image_items):
            loaded["images"].append(graph.add("LoadImage", {"image": item["path"]}, f"<Picture {index + 1}> {item['name']}"))
    if mode_name == "ref2v":
        for index, item in enumerate(shot["assets"]["videos"]):
            node = graph.add("VHS_LoadVideo", {"video": item["path"], "force_rate": 24, "custom_width": 0, "custom_height": 0, "frame_load_cap": 360, "skip_first_frames": 0, "select_every_nth": 1}, f"<Video {index + 1}> {item['name']}")
            loaded["videos"].append(node)
            loaded["video_audios"].append((node[0], 2))
        for index, item in enumerate(shot["assets"]["audios"]):
            loaded["audios"].append(graph.add("LoadAudio", {"audio": item["path"]}, f"<Audio {index + 1}> {item['name']}"))
    return loaded


def _validate_mode(settings: dict[str, Any], shot: dict[str, Any], warnings: list[str]) -> None:
    mode_name = settings["mode"]
    assets = shot["assets"]
    if mode_name == "t2v" and any(assets.values()):
        warnings.append("文生视频只使用提示词；镜头中已保存的参考素材不会接入本次生成")
    elif mode_name == "i2v":
        if not assets["images"]:
            raise DirectorValidationError("图生视频至少需要一张首帧图片；第二张图片会作为尾帧")
        if len(assets["images"]) > 2 or assets["videos"] or assets["audios"]:
            warnings.append("图生视频只接入前两张图片作为首尾帧；视频和音频参考不会接入")
    elif mode_name == "ref2v" and not any(assets.values()):
        raise DirectorValidationError("多参考生视频至少需要一项图片、视频或音频参考")

    if settings["acceleration_enabled"]:
        lora_name = settings["acceleration_lora"]
        lora = ACCELERATION_LORAS[lora_name]
        if mode_name not in lora["compatible_modes"]:
            raise DirectorValidationError(f"{lora['label']} 不兼容 {MODES[mode_name]['label']}，请更换 LoRA 或关闭加速")
        if mode_name in lora.get("experimental_modes", []):
            warnings.append(
                f"{lora['label']} 在多参考模式中属于跨模式实验用法；"
                "官方 Ref2VA 专用版本仍是 Ref2V Turbo v0.1，请对画质、声音和稳定性进行抽卡对比"
            )


def build_prompt(
    project: dict[str, Any],
    shot_id: str,
    target_dimensions_override: tuple[int, int] | None = None,
) -> BuildResult:
    project = normalize_project(project)
    try:
        shot_index = next(index for index, item in enumerate(project["shots"]) if item["id"] == shot_id)
    except StopIteration as exc:
        raise DirectorValidationError(f"找不到镜头：{shot_id}") from exc
    shot = project["shots"][shot_index]
    if not shot["prompt"].strip():
        raise DirectorValidationError(f"{shot['title']} 还没有提示词")

    settings = project["settings"]
    mode_name = settings["mode"]
    mode = MODES[mode_name]
    models = settings["models"]
    warnings: list[str] = []
    _validate_mode(settings, shot, warnings)
    target_width = int(settings["target_width"])
    target_height = int(settings["target_height"])
    target_align = 32
    if target_dimensions_override is not None:
        target_width, target_height = (int(value) for value in target_dimensions_override)
        if target_width <= 0 or target_height <= 0 or target_width % 16 or target_height % 16:
            raise DirectorValidationError("连续性目标尺寸必须是正数并按 16 对齐")
        target_align = 16
        if (target_width, target_height) != (
            int(settings["target_width"]),
            int(settings["target_height"]),
        ):
            warnings.append(
                f"检测到已有连续性链，当前镜头沿用 {target_width}×{target_height}，"
                "避免前后镜头 latent 尺寸不一致。"
            )
    inference = ACCELERATION_LORAS[settings["acceleration_lora"]] if settings["acceleration_enabled"] else DEFAULT_INFERENCE

    graph = Graph()
    model = graph.add("UNETLoader", {"unet_name": models[mode["model"]], "weight_dtype": "default"}, "① MiniMax H3 主模型")
    clip = graph.add("CLIPLoader", {"clip_name": models["clip"], "type": "minimax", "device": "default"}, "② Qwen3-VL 文本编码器")
    video_vae = graph.add("VAELoader", {"vae_name": models["video_vae"]}, "③ H3 视频 VAE")
    audio_vae = graph.add("VAELoader", {"vae_name": models["audio_vae"]}, "④ H3 音频 VAE")

    if settings["acceleration_enabled"]:
        lora = ACCELERATION_LORAS[settings["acceleration_lora"]]
        model = graph.add("LoraLoaderModelOnly", {"model": list(model), "lora_name": models[lora["model"]], "strength_model": float(settings["lora_strength"])}, f"⑤ 加速 LoRA · {lora['label']}")
    model = graph.add("PathchSageAttentionKJ", {"model": list(model), "sage_attention": "auto", "allow_compile": False}, "⑥ SageAttention · 自动")
    model = graph.add("MiniMaxH3SigmaShift", {"model": list(model), "shift_video": float(inference["shift_video"]), "shift_audio": float(inference["shift_audio"])}, f"⑦ Sigma Shift · {inference['shift_video']}/{inference['shift_audio']}")

    assets = _load_assets(graph, shot, mode_name)
    width, height = int(settings["width"]), int(settings["height"])
    base_frames = int(settings["frames"])
    requested_context = int(settings["context_frames"])
    continuity_strategy = str(settings["continuity_strategy"])
    context_span = effective_context_frames(requested_context, continuity_strategy) if settings["continuity"] else 0
    continuity_active = bool(settings["continuity"] and context_span)
    if continuity_active and context_span != requested_context:
        warnings.append(
            f"连续上下文 {requested_context} 帧不在当前策略的 H3 音视频网格上，实际使用 {context_span} 帧。"
        )

    # Native H3 lengths are 17k+5. Deliver base_frames-5 from every clip:
    # S001 drops its five-frame grid surplus, while later clips generate
    # context_span-5 extra frames before removing the complete overlap.
    frames = base_frames
    if continuity_active and shot_index > 0:
        frames += context_span - 5
    second_pass = bool(settings["second_pass"])
    upscale_scale = max(float(target_width) / width, float(target_height) / height) if second_pass else 1.0
    previous = None
    composition_anchor = None
    if continuity_active and shot_index > 0:
        context_path = f"director_console/context/{project['id']}"
        previous = graph.add(
            "H3DirectorLoadAVLatent",
            {"latent_path": context_path, "clip_index": shot_index},
            f"加载镜头 {shot_index:03d} 连续性状态",
        )
        composition_anchor = graph.add(
            "H3DirectorLatentTailFrame",
            {"latent": list(previous), "vae": list(video_vae)},
            "⑧ 上一镜头末帧 · 构图锚点",
        )
        if mode_name == "ref2v":
            asset_count = sum(len(shot["assets"][kind]) for kind in ("images", "videos", "audios"))
            if asset_count >= 12:
                raise DirectorValidationError(
                    "连续长镜头的后续多参考镜头最多使用 11 个素材，需为自动构图锚点预留 1 个位置"
                )

    generation_prompt = prompt_with_continuity_contract(
        shot["prompt"],
        shot_index,
        settings,
        anchor_picture=bool(composition_anchor and mode_name == "ref2v"),
    )

    if mode_name in {"t2v", "i2v"}:
        cond_inputs: dict[str, Any] = {"clip": list(clip), "vae": list(video_vae), "prompt": generation_prompt, "width": width, "height": height, "length": frames, "upscale_scale": upscale_scale}
        if composition_anchor is not None:
            cond_inputs["first_frame"] = list(composition_anchor)
            if mode_name == "i2v" and len(assets["images"]) > 1:
                cond_inputs["last_frame"] = list(assets["images"][1])
            if mode_name == "i2v":
                warnings.append(
                    "连续长镜头的后续片段自动使用上一段末帧作为首帧；本镜头上传的首帧不会替换连续性锚点。"
                )
        elif mode_name == "i2v":
            cond_inputs["first_frame"] = list(assets["images"][0])
            if len(assets["images"]) > 1:
                cond_inputs["last_frame"] = list(assets["images"][1])
        condition = graph.add("JZL_MiniMaxH3ImageToVideoDual", cond_inputs, "⑧ 文生/首尾帧编码")
        positive, latent, positive_high = (condition[0], 0), (condition[0], 1), (condition[0], 2)
    else:
        cond_inputs = {"clip": list(clip), "vae": list(video_vae), "audio_vae": list(audio_vae), "prompt": generation_prompt, "width": width, "height": height, "length": frames, "ref_image_size": "match"}
        image_offset = 0
        if composition_anchor is not None:
            cond_inputs["ref_images.ref_image_0"] = list(composition_anchor)
            image_offset = 1
        for index, node in enumerate(assets["images"]):
            cond_inputs[f"ref_images.ref_image_{index + image_offset}"] = list(node)
        for index, node in enumerate(assets["videos"]):
            cond_inputs[f"ref_videos.ref_video_{index}"] = list(node)
            cond_inputs[f"ref_video_audios.ref_video_audio_{index}"] = list(assets["video_audios"][index])
        for index, node in enumerate(assets["audios"]):
            cond_inputs[f"ref_audios.ref_audio_{index}"] = list(node)
        condition = graph.add("MiniMaxH3ReferenceToVideo", cond_inputs, "⑧ Ref2VA 多参考编码")
        positive, positive_high, latent = condition, condition, (condition[0], 1)

    sampler = graph.add("KSamplerSelect", {"sampler_name": str(settings["sampler"])}, "⑨ 采样器")
    noise = graph.add("RandomNoise", {"noise_seed": _seed(settings, shot, shot_index)}, "⑩ 镜头种子")
    sigmas = graph.add("BasicScheduler", {"model": list(model), "scheduler": str(settings["scheduler"]), "steps": int(settings["steps"]), "denoise": 1.0}, f"⑪ {settings['scheduler']} · {settings['steps']} 步")

    def attach_context(current_positive, current_latent):
        if not continuity_active or shot_index == 0:
            return current_positive, current_latent, None
        if previous is None:
            raise DirectorValidationError("连续性状态未加载")
        motion = graph.add(
            "H3DirectorMotionCondition",
            {
                "conditioning": list(current_positive),
                "latent": list(current_latent),
                "source_latent": list(previous),
                "context_length": context_span,
            },
            "⑫ 内置 Motion Keyframes · 保持机位/运动",
        )
        if continuity_strategy == "masked_latent":
            linked = graph.add(
                "H3DirectorGeneratedAVMaskedContext",
                {
                    "latent": list(current_latent),
                    "source_latent": list(previous),
                    "context_length": context_span,
                    "video_feather_tokens": int(settings["video_feather_tokens"]),
                    "audio_feather_ticks": int(settings["audio_feather_ticks"]),
                },
                "⑬ 内置 Masked Latent · 精确保留画面/音频",
            )
            return motion, linked, (linked[0], 1)
        return motion, current_latent, (motion[0], 1)

    trim_source = None
    if second_pass:
        if target_width * target_height <= width * height:
            raise DirectorValidationError("二次采样目标分辨率必须高于基础分辨率")
        split_at = max(1, int(settings["steps"]) - int(settings["refine_steps"]))
        split = graph.add("SplitSigmas", {"sigmas": list(sigmas), "step": split_at}, f"一采/二采分割 · {split_at}+{settings['refine_steps']}")
        guider_low = graph.add("BasicGuider", {"model": list(model), "conditioning": list(positive)}, "一采引导")
        first = graph.add("SamplerCustomAdvanced", {"noise": list(noise), "guider": list(guider_low), "sampler": list(sampler), "sigmas": [split[0], 0], "latent_image": list(latent)}, "一采 · 基础分辨率")
        separate = graph.add("LTXVSeparateAVLatent", {"av_latent": [first[0], 1]}, "拆分联合 Latent")
        upscaled = graph.add(
            "MinimaxH3LatentUpscaler3D",
            {
                "latent": [separate[0], 0],
                "model_name": models["upscaler"],
                "mode": "target dimensions",
                "mode.width": target_width,
                "mode.height": target_height,
                "align": target_align,
                "keep_proportion": False,
                "device": "cuda",
                "precision": "fp16",
            },
            "二采开关 · 3D Latent 放大",
        )
        combined = graph.add("LTXVConcatAVLatent", {"video_latent": list(upscaled), "audio_latent": [separate[0], 1]}, "重组联合 Latent")
        if mode_name == "ref2v":
            synced = graph.add("JZL_MiniMaxH3CondSync", {"positive": list(positive_high), "vae": list(video_vae), "latent": list(combined)}, "同步二采条件")
            positive_high, combined = synced, (synced[0], 1)
        positive_high, combined, trim_source = attach_context(positive_high, combined)
        guider_high = graph.add("BasicGuider", {"model": list(model), "conditioning": list(positive_high)}, "二采引导")
        sampled = graph.add("SamplerCustomAdvanced", {"noise": list(noise), "guider": list(guider_high), "sampler": list(sampler), "sigmas": [split[0], 1], "latent_image": list(combined)}, "二采 · 高分辨率细化")
    else:
        positive, latent, trim_source = attach_context(positive, latent)
        guider = graph.add("BasicGuider", {"model": list(model), "conditioning": list(positive)}, "H3 引导")
        sampled = graph.add("SamplerCustomAdvanced", {"noise": list(noise), "guider": list(guider), "sampler": list(sampler), "sigmas": list(sigmas), "latent_image": list(latent)}, "联合音视频采样")

    if continuity_active:
        graph.add("H3DirectorSaveAVLatent", {"latent": list(sampled), "filename_prefix": f"director_console/context/{project['id']}/clip", "clip_index": shot_index + 1}, f"保存镜头 {shot_index + 1:03d} 连续性状态")
    images = graph.add("VAEDecode", {"samples": list(sampled), "vae": list(video_vae)}, "解码画面")
    audio = graph.add("VAEDecodeAudio", {"samples": list(sampled), "vae": list(audio_vae)}, "解码同步音频")
    trim_frames = list(trim_source) if trim_source is not None else (5 if continuity_active else None)
    if trim_frames is not None:
        title = "裁掉上下文重叠帧" if trim_source is not None else "连续镜头时长归一化"
        trimmed = graph.add("H3DirectorTrimAV", {"images": list(images), "trim_frames": trim_frames, "audio": list(audio), "fps": int(settings["fps"]), "match_tail": True}, title)
        images, audio = trimmed, (trimmed[0], 1)
    video = graph.add("CreateVideo", {"images": list(images), "fps": int(settings["fps"]), "audio": list(audio), "bit_depth": 8}, "合成同步音视频")
    output_prefix = f"director_console/{project['id']}/{shot_index + 1:03d}_{shot['id']}"
    graph.add("SaveVideo", {"video": list(video), "filename_prefix": output_prefix, "format": "mp4", "codec": "auto"}, f"输出 · {shot['title']}")
    return BuildResult(graph.nodes, output_prefix, shot_index, mode_name, warnings)
