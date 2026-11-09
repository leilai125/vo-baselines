import os
import tempfile
from pathlib import Path


# Evaluation runs headlessly and may execute in restricted environments where
# the user's normal cache directories are read-only.
os.environ.setdefault("MPLCONFIGDIR", str(Path(tempfile.gettempdir()) / "dpvo-matplotlib"))
os.environ.setdefault("XDG_CACHE_HOME", str(Path(tempfile.gettempdir()) / "dpvo-cache"))

from .vo_eval_utils import KittiEvalOdom

__all__ = [
    "KittiEvalOdom",
]
