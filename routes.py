"""HTTP routes and persistence for the MiniMax H3 Director Console."""

from __future__ import annotations

import json
import os
import shutil
from datetime import datetime, timezone
from pathlib import Path
from typing import Any

import folder_paths
from aiohttp import web
from safetensors import safe_open
from server import PromptServer

from .workflow_builder import (
    ACCELERATION_LORAS,
    ASPECT_RATIOS,
    DEFAULT_MODELS,
    MEGAPIXELS,
    MODES,
    OUTPUT_QUALITIES,
    SEQUENCE_MODES,
    DirectorValidationError,
    build_prompt,
    normalize_project,
    slug,
)


IMAGE_EXTENSIONS = {".png", ".jpg", ".jpeg", ".webp", ".bmp"}
VIDEO_EXTENSIONS = {".mp4", ".mov", ".webm", ".avi", ".mkv"}
AUDIO_EXTENSIONS = {".wav", ".mp3", ".flac", ".ogg", ".m4a", ".aac"}
ALLOWED_EXTENSIONS = {
    "images": IMAGE_EXTENSIONS,
    "videos": VIDEO_EXTENSIONS,
    "audios": AUDIO_EXTENSIONS,
}


def _data_root() -> Path:
    getter = getattr(folder_paths, "get_user_directory", None)
    base = Path(getter()) if getter else Path(folder_paths.get_output_directory()).parent / "user"
    root = base / "director_console"
    root.mkdir(parents=True, exist_ok=True)
    return root


def _project_file(project_id: str) -> Path:
    return _data_root() / "projects" / f"{slug(project_id, 'director-project')}.json"


def _write_json_atomic(path: Path, value: dict[str, Any]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    temp = path.with_suffix(path.suffix + ".tmp")
    temp.write_text(json.dumps(value, ensure_ascii=False, indent=2), encoding="utf-8")
    os.replace(temp, path)


def _read_project(project_id: str) -> dict[str, Any] | None:
    path = _project_file(project_id)
    if not path.is_file():
        return None
    try:
        return json.loads(path.read_text(encoding="utf-8"))
    except (OSError, json.JSONDecodeError):
        return None


def _project_summaries() -> list[dict[str, Any]]:
    folder = _data_root() / "projects"
    if not folder.is_dir():
        return []
    summaries = []
    for path in folder.glob("*.json"):
        project = _read_project(path.stem)
        if not project:
            continue
        summaries.append(
            {
                "id": project.get("id", path.stem),
                "name": project.get("name", path.stem),
                "updated_at": project.get("updated_at"),
                "shot_count": len(project.get("shots") or []),
            }
        )
    return sorted(summaries, key=lambda item: item.get("updated_at") or "", reverse=True)


def _model_exists(folder_name: str, filename: str) -> bool:
    try:
        return filename in folder_paths.get_filename_list(folder_name)
    except Exception:
        return False


def _dependency_report() -> dict[str, Any]:
    import nodes

    mappings = nodes.NODE_CLASS_MAPPINGS
    model_checks = {
        "ref_model": ("diffusion_models", DEFAULT_MODELS["ref_model"]),
        "fl_model": ("diffusion_models", DEFAULT_MODELS["fl_model"]),
        "clip": ("text_encoders", DEFAULT_MODELS["clip"]),
        "video_vae": ("vae", DEFAULT_MODELS["video_vae"]),
        "audio_vae": ("vae", DEFAULT_MODELS["audio_vae"]),
        "ref_lora": ("loras", DEFAULT_MODELS["ref_lora"]),
        "fl_lora": ("loras", DEFAULT_MODELS["fl_lora"]),
        "upscaler": ("latent_upscale_models", DEFAULT_MODELS["upscaler"]),
    }
    models = {
        key: {"filename": filename, "available": _model_exists(folder, filename)}
        for key, (folder, filename) in model_checks.items()
    }
    required_nodes = [
        "MiniMaxH3ReferenceToVideo",
        "JZL_MiniMaxH3ImageToVideoDual",
        "JZL_MiniMaxH3CondSync",
        "MiniMaxH3SigmaShift",
        "LoraLoaderModelOnly",
        "PathchSageAttentionKJ",
        "H3DirectorGeneratedAVMaskedContext",
        "H3DirectorExistingVideoMaskedContext",
        "H3DirectorMaskedAVBridge",
        "H3DirectorLatentTailFrame",
        "H3DirectorMotionCondition",
        "H3DirectorTrimAV",
        "H3DirectorSaveAVLatent",
        "H3DirectorLoadAVLatent",
        "MinimaxH3LatentUpscaler3D",
        "LTXVSeparateAVLatent",
        "LTXVConcatAVLatent",
        "VHS_LoadVideo",
        "CreateVideo",
        "SaveVideo",
    ]
    optional_nodes = ["MiniMaxH3MotionContext"]
    node_report = {name: name in mappings for name in [*required_nodes, *optional_nodes]}
    return {
        "models": models,
        "nodes": node_report,
        "ready": all(node_report[name] for name in required_nodes) and all(
            item["available"] for key, item in models.items() if key not in {"ref_lora", "fl_lora", "upscaler"}
        ),
    }


def _output_items(project_id: str, shot_id: str | None = None) -> list[dict[str, Any]]:
    output_root = Path(folder_paths.get_output_directory()).resolve()
    project_folder = (output_root / "director_console" / slug(project_id, "director-project")).resolve()
    if output_root not in project_folder.parents or not project_folder.is_dir():
        return []
    items = []
    safe_shot = slug(shot_id, "") if shot_id else ""
    for path in project_folder.glob("*.mp4"):
        if safe_shot and f"_{safe_shot}" not in path.stem:
            continue
        stat = path.stat()
        items.append(
            {
                "filename": path.name,
                "subfolder": path.parent.relative_to(output_root).as_posix(),
                "type": "output",
                "size": stat.st_size,
                "modified": datetime.fromtimestamp(stat.st_mtime, timezone.utc).isoformat(),
            }
        )
    return sorted(items, key=lambda item: item["modified"], reverse=True)


def _continuity_dimensions(project_id: str, clip_index: int) -> tuple[int, int] | None:
    """Read the spatial geometry of an existing Director continuity latent."""
    if clip_index < 1:
        return None
    output_root = Path(folder_paths.get_output_directory()).resolve()
    folder = (
        output_root
        / "director_console"
        / "context"
        / slug(project_id, "director-project")
    ).resolve()
    if output_root not in folder.parents or not folder.is_dir():
        return None
    matches = sorted(folder.glob(f"*_{clip_index:05d}.safetensors"))
    if not matches:
        return None
    path = max(matches, key=lambda item: item.stat().st_mtime_ns)
    try:
        with safe_open(str(path), framework="pt", device="cpu") as handle:
            shape = tuple(handle.get_slice("video").get_shape())
    except (OSError, KeyError, ValueError):
        return None
    if len(shape) != 5 or int(shape[-2]) <= 0 or int(shape[-1]) <= 0:
        return None
    return int(shape[-1]) * 16, int(shape[-2]) * 16


def register_routes() -> None:
    routes = PromptServer.instance.routes

    @routes.get("/director_console/config")
    async def director_config(_request):
        return web.json_response(
            {
                "ok": True,
                "version": 4,
                "modes": MODES,
                "loras": ACCELERATION_LORAS,
                "aspect_ratios": ASPECT_RATIOS,
                "megapixels": MEGAPIXELS,
                "output_qualities": OUTPUT_QUALITIES,
                "sequence_modes": SEQUENCE_MODES,
                "models": DEFAULT_MODELS,
                "dependencies": _dependency_report(),
                "projects": _project_summaries(),
            }
        )

    @routes.get("/director_console/project/{project_id}")
    async def director_get_project(request):
        project = _read_project(request.match_info["project_id"])
        if project is None:
            return web.json_response({"ok": False, "error": "项目不存在"}, status=404)
        try:
            return web.json_response({"ok": True, "project": normalize_project(project)})
        except DirectorValidationError as exc:
            return web.json_response({"ok": False, "error": str(exc)}, status=400)

    @routes.post("/director_console/project")
    async def director_save_project(request):
        try:
            project = normalize_project(await request.json())
            project["updated_at"] = datetime.now(timezone.utc).isoformat()
            _write_json_atomic(_project_file(project["id"]), project)
            return web.json_response({"ok": True, "project": project})
        except (DirectorValidationError, json.JSONDecodeError) as exc:
            return web.json_response({"ok": False, "error": str(exc)}, status=400)
        except Exception as exc:
            return web.json_response({"ok": False, "error": f"保存项目失败：{exc}"}, status=500)

    @routes.post("/director_console/upload")
    async def director_upload(request):
        try:
            post = await request.post()
            upload = post.get("file")
            kind = str(post.get("kind") or "")
            project_id = slug(str(post.get("project_id") or ""), "director-project")
            shot_id = slug(str(post.get("shot_id") or ""), "shot")
            if kind not in ALLOWED_EXTENSIONS or not upload or not getattr(upload, "filename", ""):
                raise DirectorValidationError("上传参数不完整")
            filename = Path(upload.filename).name
            extension = Path(filename).suffix.lower()
            if extension not in ALLOWED_EXTENSIONS[kind]:
                raise DirectorValidationError(f"不支持的 {kind} 文件类型：{extension}")
            relative_folder = Path("director_console") / project_id / shot_id / kind
            input_root = Path(folder_paths.get_input_directory()).resolve()
            target_folder = (input_root / relative_folder).resolve()
            if input_root not in target_folder.parents:
                raise DirectorValidationError("上传路径越界")
            target_folder.mkdir(parents=True, exist_ok=True)
            stem = slug(Path(filename).stem, "asset")
            target = target_folder / f"{stem}{extension}"
            counter = 1
            while target.exists():
                target = target_folder / f"{stem}-{counter}{extension}"
                counter += 1
            with target.open("wb") as handle:
                shutil.copyfileobj(upload.file, handle, length=1024 * 1024)
            relative = target.relative_to(input_root).as_posix()
            return web.json_response(
                {
                    "ok": True,
                    "asset": {"path": relative, "name": filename, "kind": kind},
                }
            )
        except DirectorValidationError as exc:
            return web.json_response({"ok": False, "error": str(exc)}, status=400)
        except Exception as exc:
            return web.json_response({"ok": False, "error": f"上传失败：{exc}"}, status=500)

    @routes.post("/director_console/build")
    async def director_build(request):
        try:
            payload = await request.json()
            project = normalize_project(payload.get("project") or {})
            project["updated_at"] = datetime.now(timezone.utc).isoformat()
            _write_json_atomic(_project_file(project["id"]), project)
            shot_id = str(payload.get("shot_id") or "")
            shot_index = next(
                (index for index, shot in enumerate(project["shots"]) if shot["id"] == shot_id),
                -1,
            )
            target_dimensions = None
            if (
                shot_index > 0
                and project["settings"]["continuity"]
                and project["settings"]["second_pass"]
            ):
                target_dimensions = _continuity_dimensions(project["id"], shot_index)
            result = build_prompt(project, shot_id, target_dimensions)
            return web.json_response(
                {
                    "ok": True,
                    "prompt": result.prompt,
                    "output_prefix": result.output_prefix,
                    "shot_index": result.shot_index,
                    "mode": result.mode,
                    "warnings": result.warnings,
                }
            )
        except DirectorValidationError as exc:
            return web.json_response({"ok": False, "error": str(exc)}, status=400)
        except Exception as exc:
            return web.json_response({"ok": False, "error": f"构建工作流失败：{exc}"}, status=500)

    @routes.get("/director_console/outputs")
    async def director_outputs(request):
        project_id = request.query.get("project_id", "")
        shot_id = request.query.get("shot_id")
        return web.json_response({"ok": True, "outputs": _output_items(project_id, shot_id)})


register_routes()
