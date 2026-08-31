"""Validate generated prompts against a live ComfyUI node registry without sampling."""

from __future__ import annotations

import asyncio
import os
import sys
from pathlib import Path


# Comfy's node loader mirrors plugin output through the current console stream.
# Force a Unicode-safe stream so third-party nodes that print emoji cannot abort
# an otherwise unrelated registry validation on a GBK Windows console.
for _stream in (sys.stdout, sys.stderr):
    if hasattr(_stream, "reconfigure"):
        _stream.reconfigure(encoding="utf-8", errors="backslashreplace")


PLUGIN_ROOT = Path(__file__).resolve().parents[1]
COMFY_ROOT = Path(os.environ.get("COMFY_ROOT", r"D:\Comfy-Desktop\ComfyUI-Installs\comfy-Go\ComfyUI"))
sys.path.insert(0, str(PLUGIN_ROOT))
sys.path.insert(0, str(COMFY_ROOT))
sys.argv = ["main.py", "--disable-auto-launch", "--database-url", "sqlite:///:memory:"]

os.chdir(COMFY_ROOT)
import execution  # noqa: E402
import folder_paths  # noqa: E402
import main  # noqa: E402
import torch  # noqa: E402
from workflow_builder import build_prompt  # noqa: E402
from vendor.h3_masked.existing_video_extension import (  # noqa: E402
    _apply_video_context_feather,
    _seed_video_context_feather,
)
from vendor.h3_motion.patch_layout import (  # noqa: E402
    PATCH_MARKER as LAYOUT_PATCH_MARKER,
    apply_patch as apply_layout_patch,
)
from vendor.h3_motion.patch_payload import (  # noqa: E402
    MASK_PAYLOAD_MARKER,
    PATCH_MARKER as PAYLOAD_PATCH_MARKER,
    _patch_owner,
    apply_patch as apply_payload_patch,
)
import comfy.ldm.minimax.model as minimax_model  # noqa: E402
import comfy.model_base as model_base  # noqa: E402


def project(
    mode: str,
    second_pass: bool,
    image: str | None = None,
    accelerated: bool = False,
    acceleration_lora: str | None = None,
):
    assets = {"images": [], "videos": [], "audios": []}
    if image:
        assets["images"].append({"path": image, "name": image})
    return {
        "id": "integration-test",
        "settings": {
            "mode": mode,
            "aspect_ratio": "16_9",
            "megapixels": "0_6",
            "duration_seconds": 10,
            "acceleration_enabled": accelerated,
            "acceleration_lora": acceleration_lora or ("fl2v_v1_1" if mode != "ref2v" else "ref2v_v0_1"),
            "steps": 4 if accelerated else 20,
            "sampler": "euler" if accelerated else "res_multistep",
            "scheduler": "simple" if accelerated else "beta",
            "second_pass": second_pass,
            "output_quality": "720p",
            "continuity": False,
        },
        "shots": [{"id": "s001", "title": "S001", "prompt": "A cinematic shot.", "assets": assets}],
    }


def continuity_project(image: str, second_pass: bool = False):
    value = project("ref2v", second_pass, image)
    value["settings"].update(
        {
            "continuity": True,
            "continuity_strategy": "masked_latent",
            "context_frames": 39,
            "video_feather_tokens": 0,
            "audio_feather_ticks": 8,
        }
    )
    value["shots"].append(
        {
            "id": "s002",
            "title": "S002",
            "prompt": "Continue the same uninterrupted camera take.",
            "assets": value["shots"][0]["assets"],
        }
    )
    return value


async def validate(name, prompt):
    valid, error, outputs, node_errors = await execution.validate_prompt(name, prompt, None)
    if not valid:
        raise RuntimeError(f"{name}: {error}; node_errors={node_errors}")
    print(f"{name}: ok ({len(prompt)} nodes, {len(outputs)} output)")


def run():
    external_models = Path(r"C:\ComfyUI-Models")
    shared_models = Path(r"D:\Comfy-Desktop\ComfyUI-Shared\models")
    for folder_name in ("diffusion_models", "text_encoders", "vae"):
        folder_paths.add_model_folder_path(folder_name, str(external_models / folder_name))
    for folder_name in ("loras", "latent_upscale_models"):
        folder_paths.add_model_folder_path(folder_name, str(shared_models / folder_name))
    folder_paths.set_input_directory(r"D:\Comfy-Desktop\ComfyUI-Shared\input")
    folder_paths.set_output_directory(r"D:\Comfy-Desktop\ComfyUI-Shared\output")
    loop, _server, _start = main.start_comfyui()

    routes_module = next(
        module
        for name, module in sys.modules.items()
        if name.endswith("comfyui-director-console.routes")
    )
    dependency_report = routes_module._dependency_report()
    assert dependency_report["nodes"]["H3DirectorMotionCondition"]
    assert dependency_report["ready"]

    assert apply_layout_patch()
    assert apply_payload_patch()
    assert getattr(minimax_model.PackedLayout.__init__, LAYOUT_PATCH_MARKER, False)
    assert getattr(model_base.MiniMaxH3.extra_conds, PAYLOAD_PATCH_MARKER, False)

    # Reproduce the production ordering conflict without sampling: the AV-mask
    # compatibility wrapper may already surround stock MiniMaxH3.extra_conds
    # before the motion payload merge is requested. That exact wrapper is safe;
    # unknown wrappers still fail closed, and either wrapper order is idempotent.
    class DummyH3:
        pass

    def stock_extra_conds(self, **kwargs):
        return {}

    stock_extra_conds.__module__ = DummyH3.__module__

    def mask_wrapper(self, **kwargs):
        return stock_extra_conds(self, **kwargs)

    mask_wrapper.__wrapped__ = stock_extra_conds
    setattr(mask_wrapper, MASK_PAYLOAD_MARKER, True)
    DummyH3.extra_conds = mask_wrapper
    assert _patch_owner(DummyH3) == "mask_compat"

    def unknown_wrapper(self, **kwargs):
        return stock_extra_conds(self, **kwargs)

    unknown_wrapper.__wrapped__ = stock_extra_conds
    DummyH3.extra_conds = unknown_wrapper
    assert _patch_owner(DummyH3) == "foreign"

    mask_over_motion = mask_wrapper
    setattr(stock_extra_conds, PAYLOAD_PATCH_MARKER, True)
    mask_over_motion.__wrapped__ = stock_extra_conds
    DummyH3.extra_conds = mask_over_motion
    assert _patch_owner(DummyH3) == "compatible"

    mask = torch.ones((1, 1, 20, 2, 2), dtype=torch.float32)
    applied = _apply_video_context_feather(mask, 12, 3)
    assert applied == 3
    assert torch.count_nonzero(mask[:, :, :12]).item() == 0
    expected = torch.tensor([0.1464466, 0.5, 0.8535534])
    assert torch.allclose(mask[0, 0, 12:15, 0, 0], expected, atol=1e-6)
    assert torch.count_nonzero(mask[:, :, 15:] != 1.0).item() == 0

    target = torch.full((1, 1, 20, 1, 1), 10.0)
    source = torch.zeros((1, 1, 12, 1, 1))
    seeded = target.clone()
    seeded[:, :, :12] = source
    applied = _seed_video_context_feather(seeded, target, source, 12, 3)
    assert applied == 3
    assert torch.allclose(
        seeded[0, 0, 12:15, 0, 0], expected * 10.0, atol=1e-5
    )
    assert seeded[0, 0, 15, 0, 0].item() == 10.0
    image = "魂断江南-H3-firstframe-shot01.png"
    prompts = {
        "text-default": build_prompt(project("t2v", False), "s001").prompt,
        "image-v1.1": build_prompt(project("i2v", False, image, accelerated=True), "s001").prompt,
        "ref-default": build_prompt(project("ref2v", False, image), "s001").prompt,
        "ref-v1.1-experimental": build_prompt(
            project("ref2v", False, image, accelerated=True, acceleration_lora="fl2v_v1_1"),
            "s001",
        ).prompt,
        "ref-second-pass": build_prompt(project("ref2v", True, image), "s001").prompt,
        "ref-masked-continuation": build_prompt(continuity_project(image), "s002").prompt,
        "ref-masked-second-pass": build_prompt(continuity_project(image, True), "s002").prompt,
    }
    for name, prompt in prompts.items():
        loop.run_until_complete(validate(name, prompt))


if __name__ == "__main__":
    run()
