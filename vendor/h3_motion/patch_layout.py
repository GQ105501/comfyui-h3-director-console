"""Enable arbitrary MiniMax H3 keyframe anchors inside a target timeline.

Adapted from NikoDemon80/ComfyUI-H3-Motion-Context (GPL-3.0). Each motion
keyframe is presented to stock ComfyUI as a legal first-frame anchor; after
PackedLayout is built, its real frame index is restored in ``position_ids``.
Reference blocks are compensated by reading the target video's actual origin.
"""

from __future__ import annotations

import logging

import torch

import comfy.ldm.minimax.model as minimax


MC_KEY = "motion_context_index"
PATCH_MARKER = "_h3_motion_context_layout_patch"
_LOG = logging.getLogger("h3_director.motion")
_original = None
_applied = False


def _target_origin(layout) -> float:
    start, stop, kind = layout.segments[-1]
    if kind != "video" or stop <= start:
        raise RuntimeError("H3 导演台：MiniMax H3 布局尾部不再是目标视频。")
    return float(layout.position_ids[start, 0])


def _cond_time(text_len: int, latent_t: int, frame_count: int | None, index: int) -> float:
    if index == 0:
        return float(text_len)
    if frame_count is not None and index == frame_count - 1:
        return (
            float(text_len)
            + sum(minimax._video_t_spans(latent_t))
            - minimax.FRAME_RESCALE
        )
    return float(text_len) + minimax.FRAME_RESCALE * float(index)


def _fix_positions(layout, text_len, latent_t, frame_count, keyframes):
    spans = [(start, stop) for start, stop, kind in layout.segments if kind == "cond"]
    if len(spans) != len(keyframes):
        raise RuntimeError(
            f"H3 导演台：条件段数量不一致（{len(spans)} != {len(keyframes)}）。"
        )
    offset = _target_origin(layout) - float(text_len)
    for (start, stop), keyframe in zip(spans, keyframes):
        index = keyframe.get(MC_KEY)
        if index is None:
            continue
        layout.position_ids[start:stop, 0] = (
            _cond_time(text_len, latent_t, frame_count, int(index)) + offset
        )


def _patched_init(
    self,
    text_len,
    latent_t,
    latent_h,
    latent_w,
    audio_t,
    keyframes=None,
    refs=None,
    frame_count=None,
):
    _original(
        self,
        text_len,
        latent_t,
        latent_h,
        latent_w,
        audio_t,
        keyframes=keyframes,
        refs=refs,
        frame_count=frame_count,
    )
    if keyframes and any(item.get(MC_KEY) is not None for item in keyframes):
        _fix_positions(self, text_len, latent_t, frame_count, keyframes)


setattr(_patched_init, PATCH_MARKER, True)


def _patch_owner():
    cls = getattr(minimax, "PackedLayout", None)
    current = getattr(cls, "__init__", None)
    if current is None:
        return None
    if getattr(current, PATCH_MARKER, False):
        return "compatible"
    if hasattr(current, "__wrapped__"):
        return "foreign"
    home = getattr(cls, "__module__", None)
    owner = getattr(current, "__module__", None)
    return "foreign" if home and owner and home != owner else None


def _self_test():
    text_len, latent_t, latent_h, latent_w, audio_t = 7, 7, 22, 38, 16
    frame_count = sum(minimax.FRAME_PER_TOKEN[index % 5] for index in range(latent_t))

    def build(keyframes, refs=None, fix=False):
        layout = minimax.PackedLayout.__new__(minimax.PackedLayout)
        _original(
            layout,
            text_len,
            latent_t,
            latent_h,
            latent_w,
            audio_t,
            keyframes=keyframes,
            refs=refs,
            frame_count=frame_count,
        )
        if fix:
            _fix_positions(layout, text_len, latent_t, frame_count, keyframes)
        return layout

    stock = build([{"resolved_frame_index": 0}])
    marked = build([{"resolved_frame_index": 0, MC_KEY: 0}], fix=True)
    if not torch.equal(stock.position_ids, marked.position_ids):
        raise RuntimeError("首帧锚点兼容测试失败")

    run = [
        {"resolved_frame_index": 0, MC_KEY: index}
        for index in (0, 1, 5, 9)
    ]
    interior = build(run, fix=True)
    times = [
        float(interior.position_ids[start, 0])
        for start, _stop, kind in interior.segments
        if kind == "cond"
    ]
    if any(left >= right for left, right in zip(times, times[1:])):
        raise RuntimeError(f"中间帧锚点未递增：{times}")

    with_ref = build(run, refs=[{"kind": "audio", "ref_audio_t": 8}], fix=True)
    ref_times = [
        float(with_ref.position_ids[start, 0])
        for start, _stop, kind in with_ref.segments
        if kind == "cond"
    ]
    shifts = [right - left for left, right in zip(times, ref_times)]
    if not shifts or any(abs(value - shifts[0]) > 1e-6 for value in shifts):
        raise RuntimeError(f"参考块补偿不一致：{shifts}")


def apply_patch():
    global _original, _applied
    if _applied:
        return True
    if not hasattr(minimax, "PackedLayout") or not hasattr(minimax, "FRAME_RESCALE"):
        _LOG.warning("H3 Director: PackedLayout compatibility surface is unavailable")
        return False
    owner = _patch_owner()
    if owner == "compatible":
        _applied = True
        return True
    if owner == "foreign":
        _LOG.warning("H3 Director: another incompatible PackedLayout patch is active")
        return False
    _original = minimax.PackedLayout.__init__
    try:
        _self_test()
    except Exception as exc:
        _original = None
        _LOG.warning("H3 Director: layout self-test failed: %s", exc)
        return False
    minimax.PackedLayout.__init__ = _patched_init
    _applied = True
    _LOG.info("H3 Director: arbitrary motion keyframe anchors enabled")
    return True


def is_applied():
    return _applied
