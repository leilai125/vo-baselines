from __future__ import annotations

import os
from pathlib import Path
from typing import Dict, List, Optional, Sequence, Tuple, Union

import matplotlib
import numpy as np
from scipy.spatial.transform import Rotation as R

from .vo_eval_utils import KittiEvalOdom, umeyama_alignment


matplotlib.use("Agg")
from matplotlib import pyplot as plt


CameraMotion = Tuple[np.ndarray, np.ndarray]
CameraMotionPredictions = Dict[str, Sequence[CameraMotion]]
NedMotionPredictions = Dict[str, Sequence[np.ndarray]]
PathLike = Union[str, Path]
MetricValues = Dict[str, Optional[float]]
ZERO_TRANSLATION_TOLERANCE = 1e-8
NED_TO_CAMERA = np.array(
    [
        [0.0, 1.0, 0.0],
        [0.0, 0.0, 1.0],
        [1.0, 0.0, 0.0],
    ],
    dtype=np.float64,
)


def rel_pose_matrix(rotation: np.ndarray, translation: np.ndarray) -> np.ndarray:
    transform = np.eye(4, dtype=np.float64)
    transform[:3, :3] = rotation.astype(np.float64)
    transform[:3, 3] = translation.astype(np.float64)
    return transform


def pose_vector_to_matrix(pose: np.ndarray) -> np.ndarray:
    pose = np.asarray(pose, dtype=np.float64)
    if pose.shape != (6,):
        raise ValueError(f"Expected pose vector shape (6,), got {pose.shape}")
    return rel_pose_matrix(R.from_rotvec(pose[3:6]).as_matrix(), pose[:3])


def convert_ned_pose_to_camera(pose_ned: np.ndarray) -> np.ndarray:
    pose_cam = np.eye(4, dtype=np.float64)
    pose_cam[:3, :3] = NED_TO_CAMERA @ pose_ned[:3, :3] @ NED_TO_CAMERA.T
    pose_cam[:3, 3] = NED_TO_CAMERA @ pose_ned[:3, 3]
    return pose_cam


def accumulate_relative_motions(motions: Sequence[np.ndarray]) -> List[np.ndarray]:
    current_pose = np.eye(4, dtype=np.float64)
    poses = [current_pose.copy()]
    for motion in motions:
        current_pose = current_pose @ motion
        poses.append(current_pose.copy())
    return poses


def save_kitti_trajectory(poses: Sequence[np.ndarray], path: str) -> None:
    arr = np.stack([pose[:3, :].reshape(12) for pose in poses], axis=0)
    os.makedirs(os.path.dirname(path), exist_ok=True)
    np.savetxt(path, arr, fmt="%.9f")


def save_gt_trajectory_subset(pose_path: PathLike, num_frames: int, save_path: PathLike) -> None:
    poses = load_kitti_trajectory(pose_path)
    if num_frames > len(poses):
        raise ValueError(f"Requested {num_frames} GT frames from {pose_path}, but only {len(poses)} are available.")
    save_kitti_trajectory([_pose_row_to_matrix(row) for row in poses[:num_frames]], str(save_path))


def _pose_row_to_matrix(row: np.ndarray) -> np.ndarray:
    pose = np.asarray(row, dtype=np.float64).reshape(-1)
    if pose.size < 12:
        raise ValueError(f"Expected at least 12 pose values, got {pose.size}")
    transform = np.eye(4, dtype=np.float64)
    transform[:3, :] = pose[:12].reshape(3, 4)
    return transform


def load_kitti_trajectory(path: PathLike) -> np.ndarray:
    path = Path(path)
    if not path.is_file():
        raise FileNotFoundError(f"Pose file not found: {path}")

    poses = np.loadtxt(path, dtype=np.float64)
    if poses.ndim == 1:
        poses = poses[None, :]
    if poses.ndim != 2 or poses.shape[1] < 12:
        raise ValueError(f"Unexpected pose format in {path}. Expected Nx12+, got {poses.shape}.")
    return poses


def kitti_rows_to_matrices(rows: np.ndarray) -> np.ndarray:
    """Convert an N x 12 KITTI trajectory into N homogeneous poses."""
    rows = np.asarray(rows, dtype=np.float64)
    if rows.ndim == 1:
        rows = rows[None, :]
    if rows.ndim != 2 or rows.shape[1] < 12:
        raise ValueError(f"Expected an N x 12+ KITTI trajectory, got {rows.shape}")

    poses = np.repeat(np.eye(4, dtype=np.float64)[None], len(rows), axis=0)
    poses[:, :3, :] = rows[:, :12].reshape(-1, 3, 4)
    return poses


def align_trajectory_sim3(
    poses_pred: np.ndarray,
    poses_gt: np.ndarray,
) -> Optional[Tuple[np.ndarray, Dict[str, Union[float, np.ndarray]]]]:
    """Align poses with Sim(3), or return None when scale is unobservable."""
    poses_pred = np.asarray(poses_pred, dtype=np.float64)
    poses_gt = np.asarray(poses_gt, dtype=np.float64)
    if poses_pred.shape != poses_gt.shape or poses_pred.ndim != 3 or poses_pred.shape[1:] != (4, 4):
        raise ValueError(
            "Prediction and ground truth must both have shape N x 4 x 4; "
            f"got {poses_pred.shape} and {poses_gt.shape}"
        )
    if len(poses_pred) < 2:
        raise ValueError("At least two poses are required for Sim(3) alignment")

    pred_xyz = poses_pred[:, :3, 3].T
    gt_xyz = poses_gt[:, :3, 3].T
    pred_spread = np.sqrt(np.mean(np.sum(
        (pred_xyz - pred_xyz.mean(axis=1, keepdims=True)) ** 2,
        axis=0,
    )))
    gt_spread = np.sqrt(np.mean(np.sum(
        (gt_xyz - gt_xyz.mean(axis=1, keepdims=True)) ** 2,
        axis=0,
    )))
    if pred_spread <= ZERO_TRANSLATION_TOLERANCE or gt_spread <= ZERO_TRANSLATION_TOLERANCE:
        return None

    rotation, translation, scale = umeyama_alignment(pred_xyz, gt_xyz, with_scale=True)

    aligned = poses_pred.copy()
    aligned[:, :3, :3] = np.einsum("ij,njk->nik", rotation, poses_pred[:, :3, :3])
    aligned[:, :3, 3] = (
        scale * (rotation @ poses_pred[:, :3, 3].T)
        + translation[:, None]
    ).T
    return aligned, {
        "scale": float(scale),
        "rotation": rotation,
        "translation": translation,
    }


def compute_trajectory_metrics(
    poses_gt: np.ndarray,
    poses_pred: np.ndarray,
) -> MetricValues:
    """Compute the metrics used by the copied DiffVO KITTI evaluator."""
    poses_gt = np.asarray(poses_gt, dtype=np.float64)
    poses_pred = np.asarray(poses_pred, dtype=np.float64)
    if poses_gt.shape != poses_pred.shape or poses_gt.ndim != 3 or poses_gt.shape[1:] != (4, 4):
        raise ValueError(
            "Prediction and ground truth must both have shape N x 4 x 4; "
            f"got {poses_pred.shape} and {poses_gt.shape}"
        )

    evaluator = KittiEvalOdom()
    gt_dict = {index: pose for index, pose in enumerate(poses_gt)}
    pred_dict = {index: pose for index, pose in enumerate(poses_pred)}
    sequence_errors = evaluator.calc_sequence_errors(gt_dict, pred_dict)
    translation_error, rotation_error = evaluator.compute_overall_err(sequence_errors)

    # The poses have already been transformed when alignment is requested. Keep
    # alignment unset so ASE measures residual local scale drift instead of
    # being forced to zero by the legacy evaluator.
    evaluator.alignment = None
    return {
        "ave_translation": (
            float(translation_error * 100.0)
            if translation_error is not None
            else None
        ),
        "ave_rotation": (
            float(rotation_error / np.pi * 180.0 * 100.0)
            if rotation_error is not None
            else None
        ),
        "ate": float(evaluator.compute_ATE(gt_dict, pred_dict)),
        "ave_scale_error": evaluator.compute_ASE(gt_dict, pred_dict),
    }


def evaluate_trajectory_files(
    gt_path: PathLike,
    pred_path: PathLike,
    *,
    alignment: Optional[str] = None,
) -> Tuple[
    MetricValues,
    Optional[np.ndarray],
    Optional[Dict[str, Union[float, np.ndarray]]],
]:
    """Evaluate one KITTI-format trajectory and return the evaluated poses."""
    poses_gt = kitti_rows_to_matrices(load_kitti_trajectory(gt_path))
    poses_pred = kitti_rows_to_matrices(load_kitti_trajectory(pred_path))
    if len(poses_gt) != len(poses_pred):
        raise ValueError(
            f"Trajectory length mismatch for {pred_path}: "
            f"prediction has {len(poses_pred)} poses, ground truth has {len(poses_gt)}"
        )

    if alignment is None:
        evaluated_poses = poses_pred
        alignment_data: Dict[str, Union[float, np.ndarray]] = {}
    elif alignment == "7dof":
        alignment_result = align_trajectory_sim3(poses_pred, poses_gt)
        if alignment_result is None:
            return {
                "ave_translation": None,
                "ave_rotation": None,
                "ate": None,
                "ave_scale_error": None,
            }, None, None
        evaluated_poses, alignment_data = alignment_result
    else:
        raise ValueError(f"Unsupported alignment {alignment!r}; expected None or '7dof'")

    return compute_trajectory_metrics(poses_gt, evaluated_poses), evaluated_poses, alignment_data


def format_trajectory_metrics(label: str, metrics: MetricValues) -> str:
    """Format trajectory metrics consistently with model_evaluator.py."""
    translation = metrics.get("ave_translation")
    rotation = metrics.get("ave_rotation")
    ate = metrics.get("ate")
    scale_error = metrics.get("ave_scale_error")
    return (
        f"{label:<24}: "
        f"ave_translation={'N/A' if translation is None else f'{translation:.4f}'}, "
        f"ave_rotation={'N/A' if rotation is None else f'{rotation:.4f}'}, "
        f"ate={'N/A' if ate is None else f'{ate:.4f}'}, "
        f"ave_scale_error={'N/A' if scale_error is None else f'{scale_error:.4f}'}"
    )


def compute_average_trajectory_metrics(
    gt_dir: PathLike,
    pred_dir: PathLike,
    *,
    alignment: Optional[str] = None,
) -> MetricValues:
    gt_dir = Path(gt_dir)
    pred_dir = Path(pred_dir)
    metric_values: Dict[str, List[float]] = {
        "ave_translation": [],
        "ave_rotation": [],
        "ate": [],
        "ave_scale_error": [],
    }
    prediction_count = 0

    for pred_path in sorted(pred_dir.glob("*.txt")):
        prediction_count += 1
        gt_path = gt_dir / pred_path.name
        if not gt_path.is_file():
            raise FileNotFoundError(f"Missing GT trajectory for prediction {pred_path.name}: {gt_path}")
        metrics, _, _ = evaluate_trajectory_files(gt_path, pred_path, alignment=alignment)
        for key, value in metrics.items():
            if value is not None:
                metric_values[key].append(value)

    if prediction_count == 0:
        return {}
    return {
        key: float(np.mean(values)) if values else None
        for key, values in metric_values.items()
    }


def extract_xz_coordinates(poses: np.ndarray) -> Tuple[np.ndarray, np.ndarray]:
    return poses[:, 3], poses[:, 11]


def clear_prediction_dir(save_dir: str) -> None:
    """Remove previously written trajectory `.txt` predictions from a directory."""
    os.makedirs(save_dir, exist_ok=True)
    for name in os.listdir(save_dir):
        if name.endswith(".txt"):
            os.remove(os.path.join(save_dir, name))


def save_camera_motion_predictions(
    predictions: CameraMotionPredictions,
    save_dir: str,
) -> None:
    """Save camera-frame relative motions as trajectory files in KITTI pose format."""
    clear_prediction_dir(save_dir)

    for scene_id in sorted(predictions.keys()):
        camera_motions = predictions[scene_id]
        abs_poses = accumulate_relative_motions(
            [rel_pose_matrix(rotation, translation) for rotation, translation in camera_motions]
        )
        save_kitti_trajectory(abs_poses, os.path.join(save_dir, f"{scene_id}.txt"))


def save_ned_motion_predictions(
    predictions: NedMotionPredictions,
    save_dir: str,
) -> None:
    """Save NED-frame 6D motion vectors as trajectory files in KITTI pose format."""
    clear_prediction_dir(save_dir)

    for scene_id in sorted(predictions.keys()):
        motion_mats = [pose_vector_to_matrix(pose) for pose in predictions[scene_id]]
        abs_poses_ned = accumulate_relative_motions(motion_mats)
        abs_poses_cam = [convert_ned_pose_to_camera(pose_ned) for pose_ned in abs_poses_ned]
        save_kitti_trajectory(abs_poses_cam, os.path.join(save_dir, f"{scene_id}.txt"))


def save_trajectory_route_plot(
    pred_dir: PathLike,
    gt_dir: PathLike,
    *,
    sequence: str,
    method_name: str,
    color: str = "b",
    linestyle: str = "--",
    save_path: Optional[PathLike] = None,
    dpi: int = 150,
) -> str:
    """Save an X-Z trajectory plot for one sequence."""
    pred_path = Path(pred_dir) / f"{sequence}.txt"
    gt_path = Path(gt_dir) / f"{sequence}.txt"

    gt_poses = load_kitti_trajectory(gt_path)
    pred_poses = load_kitti_trajectory(pred_path)
    x_gt, z_gt = extract_xz_coordinates(gt_poses)
    x_pred, z_pred = extract_xz_coordinates(pred_poses)

    fig, ax = plt.subplots(figsize=(10, 10))
    if len(x_gt) > 0:
        ax.scatter(x_gt[0], z_gt[0], label="Start", marker="s", color="k", zorder=3)
    ax.plot(x_gt, z_gt, color="k", label="Ground truth", linewidth=2.5)
    ax.plot(x_pred, z_pred, color=color, linestyle=linestyle, label=method_name, linewidth=3.0)
    ax.set_title(f"Visual Odometry - Sequence {sequence}", fontsize=18)
    ax.set_xlabel("X [m]", fontsize=14)
    ax.set_ylabel("Z [m]", fontsize=14)
    ax.set_aspect("equal", adjustable="box")
    ax.minorticks_on()
    ax.grid(True, which="major", linestyle="-", alpha=0.6)
    ax.grid(True, which="minor", linestyle="-", alpha=0.2)
    ax.legend(loc="upper left", fontsize="large")

    route_path = Path(save_path) if save_path is not None else (Path(pred_dir) / f"route_{sequence}.png")
    route_path.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(route_path, dpi=dpi)
    plt.close(fig)
    return str(route_path)


def save_trajectory_route_plots(
    pred_dir: PathLike,
    gt_dir: PathLike,
    *,
    method_name: str,
    color: str = "b",
    linestyle: str = "--",
    dpi: int = 150,
) -> List[str]:
    """Save route plots for every prediction file in a directory."""
    pred_dir = Path(pred_dir)
    saved_paths: List[str] = []
    for pred_path in sorted(pred_dir.glob("*.txt")):
        saved_paths.append(
            save_trajectory_route_plot(
                pred_dir=pred_dir,
                gt_dir=gt_dir,
                sequence=pred_path.stem,
                method_name=method_name,
                color=color,
                linestyle=linestyle,
                dpi=dpi,
            )
        )
    return saved_paths
