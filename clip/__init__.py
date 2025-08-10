# Package initializer for CLIP_Surgery utilities.
# This subpackage provides a thin wrapper around OpenAI CLIP with
# architecture/feature "surgery" for localized explainability and for
# guiding SAM with text-derived prompts.
#
# Key references in this repository's `docs/` directory:
# - "AlignSAM – Aligning Segment Anything Model to Open Context via Reinforcement Learning" (CVPR 2024)
#   Used to guide SAM with points generated from CLIP similarity maps.
# - "Segment Anything" (SAM) (Kirillov et al., 2023)
#   Describes point-based prompting interface consumed by `similarity_map_to_points` outputs.
# - "A Closer Look at the Explainability of Contrastive Language-Image Pre-training"
#   Discusses spatial explainability of CLIP and motivates using token-level maps.
from .clip import *
from .clip import *
