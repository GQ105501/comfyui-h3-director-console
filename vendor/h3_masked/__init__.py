"""GPL-3.0 H3 masked-latent continuation components.

The implementation files in this package are vendored from
``seitanism/ComfyUI-H3-Motion-Context-MultiRef`` at commit
``87de57ba619297503fa49c9594c0c021d5b0c261``.  See ``LICENSE.upstream``.
Only the continuation classes required by the Director Console are exported.
"""

from .existing_video_extension import (
    MiniMaxH3ExistingVideoMaskedContext,
    MiniMaxH3GeneratedAVMaskedContext,
)
from .h3_masked_bridge import MiniMaxH3MaskedAVBridge

__all__ = [
    "MiniMaxH3ExistingVideoMaskedContext",
    "MiniMaxH3GeneratedAVMaskedContext",
    "MiniMaxH3MaskedAVBridge",
]

