"""Director-owned MiniMax H3 motion-conditioning support.

The compatibility shims and conditioning algorithm are adapted from
``NikoDemon80/ComfyUI-H3-Motion-Context`` at commit
``7a131a3afadc8200120f67f9236311a2c48b7445`` under GPL-3.0.  See
``NOTICE.upstream.md`` and the bundled GPL text in
``../h3_masked/LICENSE.upstream``.
"""

from .motion_condition import H3DirectorMotionCondition
from .patch_layout import apply_patch as apply_layout_patch
from .patch_payload import apply_patch as apply_payload_patch

# Install the payload merge before a masked-latent graph can lazily install its
# AV-mask wrapper. Only motion-marked graphs are changed, so unrelated H3 graphs
# retain stock behavior. The runtime classifier also supports the reverse order.
apply_payload_patch()

__all__ = [
    "H3DirectorMotionCondition",
    "apply_layout_patch",
    "apply_payload_patch",
]
