from __future__ import annotations

import os
from typing import Any, Dict, List, Optional, Tuple

import numpy as np
import torch
import torch.distributed as dist
from tqdm import tqdm

from framework.data.batch_ops import prepare_vo_inputs
from framework.engine.engine_utils import (
    TRAIN_EXTRA_SCALAR_KEYS,
    accumulate_running_stats,
    average_running_stats,
    cfg_get,
    extract_scalar_subset,
    is_dist,
    is_rank0,
)
from framework.eval.trajectory_metrics import (
    clear_prediction_dir,
    evaluate_trajectory_dir,
    save_camera_motion_predictions,
    save_gt_trajectory_subset,
    save_ned_motion_predictions,
    save_trajectory_route_plots,
)
from framework.eval.specs import TrajectoryEvalSpec
from framework.utils.pose_normalization import denormalize_pose_vectors


def _is_scale_refiner_model_type(model_type: str) -> bool:
    return model_type == "scale_refiner_model"


def _gather_predictions_rank0(pred_local: dict):
    if not is_dist():
        return pred_local

    world_size = dist.get_world_size()
    gathered = [None for _ in range(world_size)]
    dist.all_gather_object(gathered, pred_local)

    if dist.get_rank() != 0:
        return None

    merged = {}
    for data in gathered:
        for scene_id, items in data.items():
            merged.setdefault(scene_id, []).extend(items)
    return merged


class TrajectoryEvalRunner:
    def __init__(
        self,
        *,
        model: torch.nn.Module,
        loss_fn: Optional[torch.nn.Module],
        device: torch.device,
        out_dir: str,
        cfg: Any,
        logger: Optional[Any] = None,
        tensorboard_logger: Optional[Any] = None,
    ):
        self.model = model
        self.loss_fn = loss_fn
        self.device = device
        self.out_dir = out_dir
        self.cfg = cfg
        self.logger = logger
        self.tensorboard_logger = tensorboard_logger
        self.model_type = str(cfg_get(cfg, "model.type", ""))
        self.flow_scale_factor = float(cfg_get(cfg, "model.neck.args.scale_factor", 1.0))
        self.route_plot_dpi = int(
            cfg_get(
                cfg,
                "test.route_plot_dpi",
                cfg_get(
                    cfg,
                    "train.route_plot_dpi",
                    cfg_get(
                        cfg,
                        "test.kitti_route_plot_dpi",
                        cfg_get(cfg, "train.kitti_route_plot_dpi", 150),
                    ),
                ),
            )
        )

    def evaluate(self, spec: TrajectoryEvalSpec, output_tag: str, *, global_step: Optional[int] = None) -> Dict[str, float]:
        if _is_scale_refiner_model_type(self.model_type):
            return self._evaluate_scale_refiner(spec, output_tag, global_step=global_step)
        return self._evaluate_pose_model(spec, output_tag, global_step=global_step)

    def _log(self, msg: str) -> None:
        if self.logger is not None:
            self.logger.info(msg)
        else:
            print(msg)

    def _log_scalars(self, scalars: Dict[str, float], *, global_step: Optional[int], prefix: str) -> None:
        if self.tensorboard_logger is None or global_step is None:
            return
        self.tensorboard_logger.log_scalars(scalars, global_step, prefix=prefix)

    def _maybe_save_route_plots(
        self,
        spec: TrajectoryEvalSpec,
        pred_dir: str,
        gt_dir: str,
        *,
        method_name: str,
        color: str = "b",
        linestyle: str = "--",
    ) -> None:
        if not spec.save_route_plots:
            return
        save_trajectory_route_plots(
            pred_dir=pred_dir,
            gt_dir=gt_dir,
            method_name=method_name,
            color=color,
            linestyle=linestyle,
            dpi=self.route_plot_dpi,
        )

    def _materialize_gt_dir(self, spec: TrajectoryEvalSpec, save_root: str) -> str:
        if spec.gt_dir is not None:
            return spec.gt_dir

        if spec.gt_pose_paths is None or spec.gt_num_frames is None:
            raise ValueError(f"Eval spec `{spec.name}` must provide either `gt_dir` or GT subset metadata.")

        clear_prediction_dir(save_root)
        for scene, pose_path in spec.gt_pose_paths.items():
            if scene not in spec.gt_num_frames:
                raise KeyError(f"Missing GT frame count for scene {scene} in eval spec `{spec.name}`.")
            save_gt_trajectory_subset(
                pose_path,
                int(spec.gt_num_frames[scene]),
                os.path.join(save_root, f"{scene}.txt"),
            )
        return save_root

    def _update_stats_with_eval(
        self,
        stats: Dict[str, float],
        *,
        pred_dir: str,
        gt_dir: str,
        alignment: Optional[str],
        prefix: Optional[str] = None,
    ) -> Dict[str, float]:
        del alignment
        mapped = evaluate_trajectory_dir(gt_dir=gt_dir, pred_dir=pred_dir)
        if prefix is None:
            stats.update(mapped)
            return mapped

        for key, value in mapped.items():
            stats[f"{prefix}_{key}"] = value
        return mapped

    def _log_eval_summary(
        self,
        *,
        label: str,
        metrics: Dict[str, float],
        prefix: Optional[str] = None,
    ) -> None:
        key_prefix = "" if prefix is None else f"{prefix}_"
        skipped = int(metrics.get(f"{key_prefix}trans_rot_skipped_sequences", 0.0))
        skipped_suffix = f", trans_rot_skipped={skipped}" if skipped > 0 else ""
        self._log(
            f"{label} eval: "
            f"ave_translation={metrics.get(f'{key_prefix}ave_translation', 0.0):.6f}, "
            f"ave_rotation={metrics.get(f'{key_prefix}ave_rotation', 0.0):.6f}, "
            f"ate={metrics.get(f'{key_prefix}ate', 0.0):.6f}, "
            f"ave_scale_error={metrics.get(f'{key_prefix}ave_scale_error', 0.0):.6f}"
            f"{skipped_suffix}"
        )

    def _sync_after_eval(self) -> None:
        if is_dist():
            dist.barrier()

    @torch.no_grad()
    def _evaluate_pose_model(
        self,
        spec: TrajectoryEvalSpec,
        output_tag: str,
        *,
        global_step: Optional[int],
    ) -> Dict[str, float]:
        self.model.eval()

        predictions_local: Dict[str, Any] = {}

        for batch in tqdm(spec.loader, dynamic_ncols=True, disable=not is_rank0()):
            scaled_batch = prepare_vo_inputs(
                batch,
                device=self.device,
                flow_scale_factor=self.flow_scale_factor,
                require_flow=False,
            )
            outputs = self.model(scaled_batch)
            outputs["rel_pose"] = denormalize_pose_vectors(outputs["rel_pose"])

            for i in range(len(batch["scene"])):
                scene_id = batch["scene"][i]
                frame_idx = int(batch["frame_idx"][i].item())
                pose_pred = outputs["rel_pose"][i].detach().cpu().numpy()
                predictions_local.setdefault(scene_id, []).append((frame_idx, pose_pred))

        predictions = _gather_predictions_rank0(predictions_local)
        if not is_rank0():
            self._sync_after_eval()
            return {}

        for scene_id, items in predictions.items():
            uniq = {}
            for frame_idx, pose in items:
                uniq[frame_idx] = pose
            predictions[scene_id] = [uniq[k] for k in sorted(uniq)]

        stats: Dict[str, float] = {}
        save_root = os.path.join(self.out_dir, str(output_tag))
        gt_dir = self._materialize_gt_dir(spec, os.path.join(save_root, f"{spec.name}_gt"))
        save_dir = os.path.join(save_root, str(spec.save_subdir or spec.name))
        os.makedirs(save_dir, exist_ok=True)
        save_ned_motion_predictions(predictions, save_dir)
        self._maybe_save_route_plots(spec, save_dir, gt_dir, method_name="VO")
        self._update_stats_with_eval(stats, pred_dir=save_dir, gt_dir=gt_dir, alignment=spec.alignment)

        self._log_scalars(
            {
                "ave_translation": float(stats.get("ave_translation", 0.0)),
                "ave_rotation": float(stats.get("ave_rotation", 0.0)),
                "ate": float(stats.get("ate", 0.0)),
                "ave_scale_error": float(stats.get("ave_scale_error", 0.0)),
            },
            global_step=global_step,
            prefix=str(spec.tensorboard_prefix or spec.name),
        )
        self._log_eval_summary(label=spec.label, metrics=stats)
        self._sync_after_eval()
        return stats

    @torch.no_grad()
    def _evaluate_scale_refiner(
        self,
        spec: TrajectoryEvalSpec,
        output_tag: str,
        *,
        global_step: Optional[int],
    ) -> Dict[str, float]:
        self.model.eval()

        predictions_local: Dict[str, Any] = {}
        running: Dict[str, float] = {}
        n_batches = 0
        compute_loss = self.loss_fn is not None

        for batch in tqdm(spec.loader, dynamic_ncols=True, disable=not is_rank0()):
            scaled_batch = prepare_vo_inputs(
                batch,
                device=self.device,
                flow_scale_factor=self.flow_scale_factor,
                require_flow=False,
            )
            outputs = self.model(scaled_batch)
            if compute_loss:
                loss_dict: Dict[str, torch.Tensor] = self.loss_fn(outputs, scaled_batch)
                n_batches += 1
                accumulate_running_stats(running, loss_dict)

            coarse_scale = outputs["coarse_scale"].detach().cpu().numpy()
            refined_scale = outputs["refined_scale"].detach().cpu().numpy()
            rotation_eval = outputs["rotation_eval"].detach().cpu().numpy()
            direction_eval = outputs["direction_eval"].detach().cpu().numpy()

            for i in range(len(batch["scene"])):
                scene_id = batch["scene"][i]
                frame_idx = int(batch["frame_idx"][i].item())
                predictions_local.setdefault(scene_id, []).append(
                    (
                        frame_idx,
                        rotation_eval[i],
                        coarse_scale[i] * direction_eval[i],
                        refined_scale[i] * direction_eval[i],
                    )
                )

        stats = average_running_stats(running, n_batches) if compute_loss else {}

        if is_dist():
            for key, value in list(stats.items()):
                tensor = torch.tensor(value, device=self.device, dtype=torch.float32)
                dist.all_reduce(tensor, op=dist.ReduceOp.SUM)
                stats[key] = (tensor / dist.get_world_size()).item()

        predictions = _gather_predictions_rank0(predictions_local)
        if not is_rank0():
            self._sync_after_eval()
            return stats

        coarse_predictions: Dict[str, List[Tuple[np.ndarray, np.ndarray]]] = {}
        refined_predictions: Dict[str, List[Tuple[np.ndarray, np.ndarray]]] = {}

        for scene_id, items in predictions.items():
            uniq: Dict[int, Tuple[np.ndarray, np.ndarray, np.ndarray]] = {}
            for frame_idx, rotation, coarse_translation, refined_translation in items:
                uniq[int(frame_idx)] = (rotation, coarse_translation, refined_translation)

            ordered = [uniq[k] for k in sorted(uniq)]
            coarse_predictions[scene_id] = [(rotation, coarse_translation) for rotation, coarse_translation, _ in ordered]
            refined_predictions[scene_id] = [(rotation, refined_translation) for rotation, _, refined_translation in ordered]

        save_root = os.path.join(self.out_dir, str(output_tag))
        base_subdir = str(spec.save_subdir or spec.name)
        gt_dir = self._materialize_gt_dir(spec, os.path.join(save_root, f"{spec.name}_gt"))
        coarse_dir = os.path.join(save_root, f"{base_subdir}_coarse_solver")
        refined_dir = os.path.join(save_root, f"{base_subdir}_refined_solver")

        save_camera_motion_predictions(coarse_predictions, coarse_dir)
        save_camera_motion_predictions(refined_predictions, refined_dir)

        self._maybe_save_route_plots(spec, coarse_dir, gt_dir, method_name="Coarse", color="tab:blue")
        self._maybe_save_route_plots(spec, refined_dir, gt_dir, method_name="Refined", color="tab:green")
        self._update_stats_with_eval(
            stats,
            pred_dir=coarse_dir,
            gt_dir=gt_dir,
            alignment=spec.alignment,
            prefix="coarse",
        )
        refined_eval = self._update_stats_with_eval(
            stats,
            pred_dir=refined_dir,
            gt_dir=gt_dir,
            alignment=spec.alignment,
            prefix="refined",
        )
        stats.update(refined_eval)

        log_scalars = {
            "coarse_ave_translation": float(stats.get("coarse_ave_translation", 0.0)),
            "coarse_ave_rotation": float(stats.get("coarse_ave_rotation", 0.0)),
            "coarse_ate": float(stats.get("coarse_ate", 0.0)),
            "coarse_ave_scale_error": float(stats.get("coarse_ave_scale_error", 0.0)),
            "refined_ave_translation": float(stats.get("refined_ave_translation", 0.0)),
            "refined_ave_rotation": float(stats.get("refined_ave_rotation", 0.0)),
            "refined_ate": float(stats.get("refined_ate", 0.0)),
            "refined_ave_scale_error": float(stats.get("refined_ave_scale_error", 0.0)),
        }
        log_scalars.update(extract_scalar_subset(stats, ("loss", *TRAIN_EXTRA_SCALAR_KEYS)))

        self._log_scalars(log_scalars, global_step=global_step, prefix=str(spec.tensorboard_prefix or spec.name))
        self._log_eval_summary(label=f"{spec.label} coarse", metrics=stats, prefix="coarse")
        self._log_eval_summary(label=f"{spec.label} refined", metrics=stats, prefix="refined")
        self._sync_after_eval()
        return stats
