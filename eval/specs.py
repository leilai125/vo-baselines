from __future__ import annotations

from dataclasses import dataclass
from typing import Any, Dict, Optional


@dataclass(frozen=True)
class TrajectoryEvalSpec:
    name: str
    label: str
    loader: Any
    gt_dir: Optional[str] = None
    gt_pose_paths: Optional[Dict[str, str]] = None
    gt_num_frames: Optional[Dict[str, int]] = None
    save_subdir: Optional[str] = None
    tensorboard_prefix: Optional[str] = None
    alignment: Optional[str] = None
    save_route_plots: bool = False

