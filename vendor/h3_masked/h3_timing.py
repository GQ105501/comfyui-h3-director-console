"""Small MiniMax H3 temporal-grid helpers shared by the masking/streaming nodes."""

FPS = 24
AUDIO_LATENT_HZ = 40


def largest_h3_video_run(frames: int) -> int:
    """Largest exact full-run video-VAE input <= frames: 5, 22, 39, ..."""
    n = int(frames)
    if n < 5:
        return 0
    return 5 + ((n - 5) // 17) * 17


def is_h3_video_run(frames: int) -> bool:
    """True for native full H3 video-VAE runs (5, 22, 39, ...)."""
    n = int(frames)
    return n >= 5 and (n - 5) % 17 == 0


def is_exact_av_boundary(frames: int) -> bool:
    """True when a 24-fps frame boundary is integral on H3's 40-Hz audio grid."""
    return (int(frames) * AUDIO_LATENT_HZ) % FPS == 0


def video_runs_through(limit: int = 243):
    n = 5
    out = []
    while n <= int(limit):
        out.append(n)
        n += 17
    return out


def preferred_av_runs_through(limit: int = 243):
    """Native H3 runs whose endpoint also lands exactly on the audio clock."""
    return [n for n in video_runs_through(limit) if is_exact_av_boundary(n)]



def sample_boundary_from_frames(frame_position: int, sample_rate: int, fps: int = FPS) -> int:
    """Nearest PCM sample on the absolute video timeline for a frame boundary."""
    return int(round(int(frame_position) / float(fps) * int(sample_rate)))


def sample_boundary_from_seconds(seconds: float, sample_rate: int) -> int:
    """Nearest PCM sample on an absolute time boundary."""
    return int(round(float(seconds) * int(sample_rate)))


def sample_span_for_frame_interval(start_frame: int, frame_count: int, sample_rate: int, fps: int = FPS) -> int:
    """PCM samples covered by an absolute frame interval without relative-rounding drift."""
    start = sample_boundary_from_frames(start_frame, sample_rate, fps)
    end = sample_boundary_from_frames(int(start_frame) + int(frame_count), sample_rate, fps)
    return end - start

def crossfade_plan(context_frames: int, requested_crossfade: int):
    """Return (frames_to_drop_before_blend, effective_overlap).

    The retained overlap is always the *last* matching context frames, so a
    shorter visual crossfade never replays the older part of a longer context.
    """
    n = max(0, int(context_frames))
    effective = min(n, max(0, int(requested_crossfade)))
    return n - effective, effective
