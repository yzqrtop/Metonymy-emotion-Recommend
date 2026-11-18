"""MetonymyRec: metonymy-aware multimodal recommendation (paper reproduction).

Package layout
--------------
data        dataset schema, seed builder, loaders, tokenisation utilities
annotation  Qwen2.5-VL-72B / offline annotation pipeline with confidence gating
models      encoders, dual-path gated localisation, prototype prior, decoder
training    two-stage training (MTL encoder, soft-prefix decoder)
evaluation  metrics, business-level / user-level / transfer / sentiment protocols
"""

import os
import sys

# Optional isolated wheels used when the system Python is read-only.
_VENDOR = os.path.join(os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__)))), ".vendor")
if os.path.isdir(_VENDOR) and _VENDOR not in sys.path:
    sys.path.insert(0, _VENDOR)

__version__ = "2.0.0"
