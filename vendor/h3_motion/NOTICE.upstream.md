# Upstream notice

The motion-conditioning implementation in this directory is adapted from
[NikoDemon80/ComfyUI-H3-Motion-Context](https://github.com/NikoDemon80/ComfyUI-H3-Motion-Context),
commit `7a131a3afadc8200120f67f9236311a2c48b7445`.

Copyright (C) 2026 NikoDemon80. Licensed under GNU GPL version 3. The complete
license text is bundled at `../h3_masked/LICENSE.upstream`.

Director Console modifications made on 2026-08-27:

- reduced the public surface to one latent-to-conditioning node;
- retained the arbitrary H3 keyframe layout and Ref2VA payload fixes;
- removed decoded-frame, external-plugin, audio-reference, and checkpoint nodes;
- added strict latent-grid validation for director-managed clip chaining.
