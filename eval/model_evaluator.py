from __future__ import annotations

from pathlib import Path
from typing import Any, Dict, Iterable, List, Sequence, Tuple

import numpy as np
import torch
import torch.distributed as dist
from tqdm import tqdm

from framework.eval.vo_eval_utils import (
    compute_metrics,
    save_camera_motion_predictions,
    save_ned_motion_predictions,
    save_trajectory_route_plots,
)
from framework.utils.pose_normalization import denormalize_pose_vectors


def prepare_vo_inputs(
    batch: Dict[str, Any],
    device: torch.device,
) -> Dict[str, Any]:
    input_keys = (
        "image1",
        "image2",
        "rel_pose",
        "depth1",
        "depth2",
        "source_intr_layer",
        "target_intr_layer",
    )
    inputs = {
        key: batch[key].to(device, non_blocking=True)
        for key in input_keys
    }
    for key in ("depth1_valid_mask", "depth2_valid_mask"):
        if key in batch:
            inputs[key] = batch[key].to(device, non_blocking=True)
    return inputs


def _is_distributed() -> bool:
    return dist.is_available() and dist.is_initialized()


def _is_rank0() -> bool:
    return not _is_distributed() or dist.get_rank() == 0


def _gather_predictions(predictions_local: Dict[str, List[Any]]) -> Dict[str, List[Any]] | None:
    if not _is_distributed():
        return predictions_local

    gathered = [None for _ in range(dist.get_world_size())]
    dist.all_gather_object(gathered, predictions_local)
    if not _is_rank0():
        return None

    predictions: Dict[str, List[Any]] = {}
    for rank_predictions in gathered:
        for scene_id, scene_predictions in rank_predictions.items():
            predictions.setdefault(scene_id, []).extend(scene_predictions)
    return predictions


def _sort_predictions(
    predictions: Dict[str, List[Any]],
) -> Dict[str, List[Any]]:
    sorted_predictions: Dict[str, List[Any]] = {}
    for scene_id, scene_predictions in predictions.items():
        predictions_by_frame = {
            int(prediction[0]): prediction
            for prediction in scene_predictions
        }
        sorted_predictions[scene_id] = [
            predictions_by_frame[frame_idx]
            for frame_idx in sorted(predictions_by_frame)
        ]
    return sorted_predictions


def _collect_pose_predictions(
    outputs: Dict[str, Any],
    batch: Dict[str, Any],
    predictions: Dict[str, List[Any]],
) -> None:
    relative_poses = denormalize_pose_vectors(outputs["rel_pose"])
    for index, scene_id in enumerate(batch["scene"]):
        frame_idx = int(batch["frame_idx"][index].item())
        relative_pose = relative_poses[index].detach().cpu().numpy()
        predictions.setdefault(scene_id, []).append((frame_idx, relative_pose))


def _collect_scale_predictions(
    outputs: Dict[str, Any],
    batch: Dict[str, Any],
    predictions: Dict[str, List[Any]],
) -> None:
    rotations = outputs["rotation_eval"].detach().cpu().numpy()
    directions = outputs["direction_eval"].detach().cpu().numpy()
    coarse_scales = outputs["coarse_scale"].detach().cpu().numpy()
    refined_scales = outputs["refined_scale"].detach().cpu().numpy()

    for index, scene_id in enumerate(batch["scene"]):
        frame_idx = int(batch["frame_idx"][index].item())
        coarse_translation = coarse_scales[index] * directions[index]
        refined_translation = refined_scales[index] * directions[index]
        predictions.setdefault(scene_id, []).append(
            (
                frame_idx,
                rotations[index],
                coarse_translation,
                refined_translation,
            )
        )


def _prefix_metrics(
    metrics: Dict[str, float],
    prefix: str,
) -> Dict[str, float]:
    return {
        f"{prefix}_{metric_name}": value
        for metric_name, value in metrics.items()
    }


def format_trajectory_metrics(
    label: str,
    metrics: Dict[str, float],
    *,
    prefix: str = "",
) -> str:
    metric_prefix = f"{prefix}_" if prefix else ""
    translation = metrics.get(f"{metric_prefix}ave_translation", 0.0)
    rotation = metrics.get(f"{metric_prefix}ave_rotation", 0.0)
    ate = metrics.get(f"{metric_prefix}ate", 0.0)
    scale_error = metrics.get(f"{metric_prefix}ave_scale_error", 0.0)
    return (
        f"{label:<20}: "
        f"ave_translation={translation:.4f}, "
        f"ave_rotation={rotation:.4f}, "
        f"ate={ate:.4f}, "
        f"ave_scale_error={scale_error:.4f}"
    )


def format_scale_refiner_metrics(
    label: str,
    metrics: Dict[str, float],
) -> str:
    parts = [
        f"{label:<20}: "
        f"translation={metrics.get('refined_ave_translation', 0.0):.4f}",
        f"rotation={metrics.get('refined_ave_rotation', 0.0):.4f}",
        f"ate={metrics.get('refined_ate', 0.0):.4f}",
        f"coarse_scale_error={metrics.get('coarse_ave_scale_error', 0.0):.4f}",
        f"refined_scale_error={metrics.get('refined_ave_scale_error', 0.0):.4f}",
    ]
    if "fb_consistency_ratio" in metrics:
        parts.append(
            f"fb_consistency={metrics['fb_consistency_ratio']:.4f}"
        )
    if "valid_support_ratio" in metrics:
        parts.append(
            f"valid_support={metrics['valid_support_ratio']:.4f}"
        )
    return ", ".join(parts)


def _accumulate_mask_ratio(
    totals: torch.Tensor,
    *,
    offset: int,
    value: torch.Tensor,
) -> None:
    totals[offset] += value.float().sum()
    totals[offset + 1] += value.numel()


def _evaluate_pose_predictions(
    predictions: Dict[str, List[Any]],
    *,
    save_root: Path,
    gt_dir: str,
    save_route_plots: bool,
) -> Dict[str, float]:
    pose_predictions: Dict[str, Sequence[np.ndarray]] = {
        scene_id: [prediction[1] for prediction in scene_predictions]
        for scene_id, scene_predictions in predictions.items()
    }
    save_ned_motion_predictions(pose_predictions, str(save_root))
    if save_route_plots:
        save_trajectory_route_plots(str(save_root), gt_dir, "VO")
    return compute_metrics(gt_dir, str(save_root))


def _evaluate_scale_predictions(
    predictions: Dict[str, List[Any]],
    *,
    save_root: Path,
    gt_dir: str,
    save_route_plots: bool,
) -> Dict[str, float]:
    coarse_predictions: Dict[str, List[Tuple[np.ndarray, np.ndarray]]] = {}
    refined_predictions: Dict[str, List[Tuple[np.ndarray, np.ndarray]]] = {}

    for scene_id, scene_predictions in predictions.items():
        coarse_predictions[scene_id] = [
            (rotation, coarse_translation)
            for _, rotation, coarse_translation, _ in scene_predictions
        ]
        refined_predictions[scene_id] = [
            (rotation, refined_translation)
            for _, rotation, _, refined_translation in scene_predictions
        ]

    coarse_dir = save_root / "coarse_solver"
    refined_dir = save_root / "refined_solver"
    save_camera_motion_predictions(coarse_predictions, str(coarse_dir))
    save_camera_motion_predictions(refined_predictions, str(refined_dir))

    if save_route_plots:
        save_trajectory_route_plots(
            str(coarse_dir),
            gt_dir,
            "Coarse solver",
            color="tab:blue",
        )
        save_trajectory_route_plots(
            str(refined_dir),
            gt_dir,
            "Refined solver",
            color="tab:green",
        )

    coarse_metrics = compute_metrics(gt_dir, str(coarse_dir))
    refined_metrics = compute_metrics(gt_dir, str(refined_dir))
    metrics = _prefix_metrics(coarse_metrics, "coarse")
    metrics.update(_prefix_metrics(refined_metrics, "refined"))
    metrics.update(refined_metrics)
    return metrics


@torch.no_grad()
def evaluate_model(
    *,
    model: torch.nn.Module,
    loader: Iterable,
    dataset_name: str,
    output_tag: str,
    device: torch.device,
    out_dir: str,
    cfg: Dict[str, Any],
) -> Dict[str, float]:
    model.eval()
    model_type = str(cfg["model"]["type"])
    is_scale_refiner = model_type == "scale_refiner_model"
    predictions_local: Dict[str, List[Any]] = {}
    diagnostic_totals = torch.zeros(4, device=device, dtype=torch.float64)
    fb_enabled = bool(
        cfg["model"].get("fb_consistency", {}).get("enabled", False)
    )

    progress_loader = tqdm(
        loader,
        dynamic_ncols=True,
        disable=not _is_rank0(),
    )
    for batch in progress_loader:
        inputs = prepare_vo_inputs(batch, device=device)
        outputs = model(inputs)
        if is_scale_refiner:
            _collect_scale_predictions(outputs, batch, predictions_local)
            if fb_enabled:
                _accumulate_mask_ratio(
                    diagnostic_totals,
                    offset=0,
                    value=outputs["fb_consistency_mask"],
                )
            _accumulate_mask_ratio(
                diagnostic_totals,
                offset=2,
                value=outputs["refined_valid_mask"],
            )
        else:
            _collect_pose_predictions(outputs, batch, predictions_local)

    if _is_distributed():
        dist.all_reduce(diagnostic_totals, op=dist.ReduceOp.SUM)
    gathered_predictions = _gather_predictions(predictions_local)
    if not _is_rank0():
        return {}
    if gathered_predictions is None:
        raise RuntimeError("Rank 0 did not receive evaluation predictions")

    predictions = _sort_predictions(gathered_predictions)
    gt_dir = cfg["data"]["eval_sets"][f"{dataset_name}_args"]["pose_root"]
    save_root = Path(out_dir) / str(output_tag) / dataset_name
    save_route_plots = dataset_name == "kitti"

    if is_scale_refiner:
        metrics = _evaluate_scale_predictions(
            predictions,
            save_root=save_root,
            gt_dir=gt_dir,
            save_route_plots=save_route_plots,
        )
        if fb_enabled and diagnostic_totals[1].item() > 0:
            metrics["fb_consistency_ratio"] = (
                diagnostic_totals[0] / diagnostic_totals[1]
            ).item()
        if diagnostic_totals[3].item() > 0:
            metrics["valid_support_ratio"] = (
                diagnostic_totals[2] / diagnostic_totals[3]
            ).item()
        return metrics
    return _evaluate_pose_predictions(
        predictions,
        save_root=save_root,
        gt_dir=gt_dir,
        save_route_plots=save_route_plots,
    )
