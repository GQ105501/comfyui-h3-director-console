"""Keep MiniMax H3 motion keyframes when Ref2VA references are present.

Adapted from NikoDemon80/ComfyUI-H3-Motion-Context (GPL-3.0). Stock ComfyUI
builds keyframe video latents and then overwrites them with reference-video
latents. The wrapper concatenates both lists in the same order as PackedLayout.
Unmarked graphs retain byte-for-byte stock behaviour.
"""

from __future__ import annotations

import logging

import comfy.model_base as model_base


MC_KEY = "motion_context_index"
PATCH_MARKER = "_h3_motion_context_payload_patch"
MASK_PAYLOAD_MARKER = "_h3_existing_video_av_mask_payload_compat_v2"
_LOG = logging.getLogger("h3_director.motion")
_original = None
_applied = False


def _patched_extra_conds(self, **kwargs):
    out = _original(self, **kwargs)
    keyframes = kwargs.get("minimax_keyframes")
    refs = kwargs.get("minimax_refs")
    if not keyframes or not refs or not any(MC_KEY in item for item in keyframes):
        return out

    cond = out.get("minimax_payload")
    payload = getattr(cond, "cond", None) if cond is not None else None
    if not isinstance(payload, dict):
        raise RuntimeError("H3 导演台：无法访问 MiniMax H3 条件 payload。")

    payload["cond_video_latents"] = (
        [item["latent"] for item in keyframes if "latent" in item]
        + [item["latent"] for item in refs if "latent" in item]
    )
    payload["cond_audio_latents"] = [
        item["audio_latent"]
        for item in refs
        if item.get("audio_latent") is not None
    ]
    frame_count = kwargs.get("minimax_frame_count")
    if frame_count is not None:
        payload["frame_count"] = frame_count
    return out


setattr(_patched_extra_conds, PATCH_MARKER, True)


def _wrapper_chain(function):
    seen = set()
    while function is not None and id(function) not in seen:
        seen.add(id(function))
        yield function
        function = getattr(function, "__wrapped__", None)


def _patch_owner(cls):
    current = getattr(cls, "extra_conds", None)
    if current is None:
        return None
    chain = list(_wrapper_chain(current))
    if any(getattr(function, PATCH_MARKER, False) for function in chain):
        return "compatible"
    mask_wrappers = [
        function
        for function in chain
        if getattr(function, MASK_PAYLOAD_MARKER, False)
    ]
    unknown_wrappers = [
        function
        for function in chain
        if hasattr(function, "__wrapped__")
        and not getattr(function, MASK_PAYLOAD_MARKER, False)
    ]
    if unknown_wrappers:
        return "foreign"
    if mask_wrappers:
        # The masked-latent wrapper only adds video/audio denoise-mask outputs.
        # It deliberately calls its wrapped base first, so the motion/ref merge
        # can safely sit either inside or outside it. This exact marker is part
        # of our vendored h3_masked package; no unnamed wrapper is trusted.
        return "mask_compat"
    home = getattr(cls, "__module__", None)
    owner = getattr(current, "__module__", None)
    return "foreign" if home and owner and home != owner else None


def apply_patch():
    global _original, _applied
    if _applied:
        return True
    cls = getattr(model_base, "MiniMaxH3", None)
    if cls is None or not hasattr(cls, "extra_conds"):
        _LOG.warning("H3 Director: MiniMaxH3.extra_conds is unavailable")
        return False
    owner = _patch_owner(cls)
    if owner == "compatible":
        _applied = True
        return True
    if owner == "foreign":
        _LOG.warning("H3 Director: another incompatible payload patch is active")
        return False
    _original = cls.extra_conds
    cls.extra_conds = _patched_extra_conds
    _applied = True
    _LOG.info(
        "H3 Director: Ref2VA and motion keyframe payloads can coexist%s",
        " alongside AV masks" if owner == "mask_compat" else "",
    )
    return True


def is_applied():
    return _applied
