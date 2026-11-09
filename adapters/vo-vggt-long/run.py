"""Run unchanged VGGT-Long inference and evaluate prepared VO datasets."""

from __future__ import annotations

import argparse
import builtins
from collections.abc import Iterable, Mapping
from concurrent.futures import ThreadPoolExecutor, as_completed
from contextlib import contextmanager
import importlib
import json
import os
from pathlib import Path
import signal
import subprocess
import sys
import tempfile
import threading
import traceback
from typing import TYPE_CHECKING

if TYPE_CHECKING:
    import numpy as np


PROJECT_ROOT = Path(__file__).resolve().parents[2]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))
IMAGE_SUFFIXES = {".png", ".jpg", ".jpeg"}
_CANCEL = threading.Event()
_PROCESS_LOCK = threading.Lock()
_PROCESSES = set()


METRIC_NAMES = ("ave_translation", "ave_rotation", "ate", "ave_scale_error")
ALIGNMENTS = ("before_alignment", "after_alignment")


def validate_poses(poses: np.ndarray, label: str = "poses") -> None:
    """Reject invalid transforms without changing a baseline's predictions."""
    import numpy as np

    if poses.ndim != 3 or poses.shape[1:] != (4, 4) or len(poses) == 0:
        raise ValueError(f"{label}: expected nonempty N x 4 x 4 poses; got {poses.shape}")
    if not np.isfinite(poses).all():
        raise ValueError(f"{label}: poses contain nonfinite values")
    if not np.allclose(poses[:, 3, :], [0.0, 0.0, 0.0, 1.0], atol=1e-6):
        raise ValueError(f"{label}: invalid homogeneous bottom row")
    rotations = poses[:, :3, :3]
    if not (
        np.allclose(rotations.transpose(0, 2, 1) @ rotations, np.eye(3), atol=5e-3)
        and np.allclose(np.linalg.det(rotations), 1.0, atol=5e-3)
    ):
        raise ValueError(f"{label}: rotations are not proper rigid rotations")


def load_poses(path: str | Path, columns: int | None = None) -> np.ndarray:
    """Read an exact 12-column GT/export or 16-column native C2W trajectory."""
    import numpy as np

    path = Path(path)
    rows = np.loadtxt(path, dtype=np.float64, ndmin=2)
    expected = (columns,) if columns is not None else (12, 16)
    if columns is not None and columns not in (12, 16):
        raise ValueError(f"Unsupported pose width {columns}; expected 12 or 16")
    if rows.ndim != 2 or len(rows) == 0 or rows.shape[1] not in expected:
        raise ValueError(f"{path}: expected nonempty N x {expected} pose rows; got {rows.shape}")
    if rows.shape[1] == 16:
        poses = rows.reshape(-1, 4, 4)
    else:
        poses = np.repeat(np.eye(4)[None], len(rows), axis=0)
        poses[:, :3, :] = rows.reshape(-1, 3, 4)
    validate_poses(poses, str(path))
    return poses


def rebase_c2w(native: np.ndarray, first_gt_pose: np.ndarray) -> np.ndarray:
    """Fix only the initial rigid gauge, retaining native prediction scale."""
    import numpy as np

    native = np.asarray(native, dtype=np.float64)
    first_gt_pose = np.asarray(first_gt_pose, dtype=np.float64)
    validate_poses(native, "native C2W")
    validate_poses(first_gt_pose[None], "first GT pose")
    return (first_gt_pose @ np.linalg.inv(native[0]))[None] @ native


def clean_metrics(metrics: Mapping) -> dict[str, float | None]:
    """Use strict JSON-compatible numbers; zero remains a valid metric."""
    import numpy as np

    result = {}
    for name in METRIC_NAMES:
        value = metrics.get(name)
        result[name] = float(value) if value is not None and np.isfinite(value) else None
    return result


def evaluate_scene(
    native_path: str | Path,
    gt_path: str | Path,
    output_dir: str | Path,
    scene: str,
) -> dict:
    """Write both pose exports and evaluate them using the unchanged evaluator."""
    from eval.trajectory_metrics import evaluate_trajectory_files, save_kitti_trajectory

    output_dir = Path(output_dir)
    before_path = output_dir / "before_alignment" / f"{scene}.txt"
    after_path = output_dir / "after_alignment" / f"{scene}.txt"
    # Re-running a scene must not leave an earlier aligned trajectory behind if
    # this run's alignment is unavailable or pose validation fails.
    before_path.unlink(missing_ok=True)
    after_path.unlink(missing_ok=True)

    gt = load_poses(gt_path, columns=12)
    native = load_poses(native_path)
    if len(native) != len(gt):
        raise ValueError(
            f"{scene}: prediction/GT count mismatch: {len(native)} versus {len(gt)}"
        )
    converted = rebase_c2w(native, gt[0])
    save_kitti_trajectory(converted, str(before_path))
    before_metrics, _, _ = evaluate_trajectory_files(gt_path, before_path, alignment=None)

    record = {
        "num_poses": len(converted),
        "before_alignment": clean_metrics(before_metrics),
        "after_alignment": {name: None for name in METRIC_NAMES},
    }
    if len(converted) < 2:
        record["alignment_unavailable_reason"] = "Sim(3) requires at least two poses"
        return record

    after_metrics, aligned, _ = evaluate_trajectory_files(
        gt_path, before_path, alignment="7dof"
    )
    record["after_alignment"] = clean_metrics(after_metrics)
    if aligned is None:
        record["alignment_unavailable_reason"] = (
            "Sim(3) scale is unobservable for a stationary GT or prediction trajectory"
        )
    else:
        validate_poses(aligned, "Sim(3)-aligned poses")
        save_kitti_trajectory(aligned, str(after_path))
    return record


def aggregate_scene_metrics(scene_records: Mapping | Iterable[Mapping]) -> dict:
    """Compute independent per-metric means over available finite scene values."""
    import numpy as np

    records = list(scene_records.values() if isinstance(scene_records, Mapping) else scene_records)
    averages = {}
    counts = {}
    for alignment in ALIGNMENTS:
        averages[alignment] = {}
        counts[alignment] = {}
        for name in METRIC_NAMES:
            values = []
            for record in records:
                group = record.get(alignment, {})
                value = group.get(name)
                if value is not None and np.isfinite(value):
                    values.append(float(value))
            averages[alignment][name] = float(np.mean(values)) if values else None
            counts[alignment][name] = len(values)
    return {"averages": averages, "valid_scene_counts": counts}


def has_ground_truth(path):
    if not path.is_file():
        return False
    with path.open() as stream:
        return any(line.strip() and not line.lstrip().startswith("#") for line in stream)


def discover_scenes(prepared_root, image_subdir="image_2"):
    prepared_root = Path(prepared_root)
    sequences, poses = prepared_root / "sequences", prepared_root / "poses"
    if not sequences.is_dir() or not poses.is_dir():
        raise FileNotFoundError(f"Expected sequences/ and poses/ under {prepared_root}")
    scenes, skipped = [], []
    for directory in sorted(sequences.iterdir()):
        if not directory.is_dir():
            continue
        gt_path = poses / f"{directory.name}.txt"
        if not has_ground_truth(gt_path):
            skipped.append(directory.name)
            continue
        scenes.append({"scene": directory.name,
                       "image_dir": str(directory / image_subdir),
                       "gt_path": str(gt_path)})
    return scenes, skipped


def split_scenes(scenes, devices):
    if not devices:
        raise ValueError("Select at least one GPU")
    return [(device, scenes[index::len(devices)]) for index, device in enumerate(devices)]


def resolve_devices(spec, visible=None):
    try:
        indices = [int(value.strip()) for value in spec.split(",")]
    except ValueError as exc:
        raise ValueError("--gpus must be comma-separated GPU indices, e.g. 0,1,2,3") from exc
    if not indices or min(indices) < 0 or len(set(indices)) != len(indices):
        raise ValueError("GPU indices must be nonnegative and unique")
    if visible is None:
        return [str(index) for index in indices]
    available = [value.strip() for value in visible.split(",") if value.strip()]
    if not available or "-1" in available or max(indices) >= len(available):
        raise ValueError("Requested GPUs are outside the existing CUDA_VISIBLE_DEVICES mask")
    return [available[index] for index in indices]


def load_settings(path):
    import yaml

    with Path(path).open() as stream:
        config = yaml.safe_load(stream)
    if not isinstance(config, dict) or not isinstance(config.get("datasets"), dict):
        raise ValueError("Configuration must contain an ordered datasets mapping")

    def absolute(value):
        value = Path(value).expanduser()
        return value.resolve() if value.is_absolute() else (PROJECT_ROOT / value).resolve()

    data_root = absolute(config.get("dataset_root", "/data/lei/vo_benchmarks"))
    return {
        "baseline_dir": absolute(config.get("baseline_dir", "vo-vggt-long")),
        "baseline_config": absolute(config.get("baseline_config", "vo-vggt-long/configs/base_config.yaml")),
        "checkpoint": absolute(config["checkpoint"]) if config.get("checkpoint") else None,
        "python_executable": str(config.get("python_executable") or sys.executable),
        "scratch_root": absolute(config.get("scratch_root", "/tmp/vo-vggt-long")),
        "output_root": absolute(config.get("output_root", "results/vo-vggt-long")),
        "image_subdir": config.get("image_subdir", "image_2"),
        "datasets": {name: (Path(root) if Path(root).is_absolute() else data_root / root)
                     for name, root in config["datasets"].items()},
    }


@contextmanager
def pose_only_exports(module):
    """Suppress export-only PLY writers, without replacing any estimation method."""
    original_writer = module.save_confident_pointcloud_batch
    had_open = "open" in vars(module)
    original_open = vars(module).get("open", builtins.open)

    def output_open(file, mode="r", *args, **kwargs):
        if isinstance(file, (str, os.PathLike)) and (
            str(file).lower().endswith(".ply") or Path(file).name == "intrinsic.txt"
        ):
            if any(flag in mode for flag in "wax+"):
                return original_open(os.devnull, mode, *args, **kwargs)
        return original_open(file, mode, *args, **kwargs)

    module.save_confident_pointcloud_batch = lambda *args, **kwargs: None
    module.open = output_open
    try:
        yield
    finally:
        module.save_confident_pointcloud_batch = original_writer
        if had_open:
            module.open = original_open
        else:
            del module.open


def scene_worker(task_path):
    """One fresh process for one scene; visibility is set before Torch imports."""
    with Path(task_path).open() as stream:
        task = json.load(stream)
    output = Path(task["output_dir"])
    scene = task["scene"]
    for group in ("before_alignment", "after_alignment"):
        (output / group / f"{scene}.txt").unlink(missing_ok=True)
    try:
        images = sorted(path for path in Path(task["image_dir"]).iterdir()
                        if path.is_file() and path.suffix.lower() in IMAGE_SUFFIXES)
        gt = load_poses(task["gt_path"], columns=12)
        if not images or len(images) != len(gt):
            raise ValueError(f"{scene}: image/GT count mismatch: {len(images)} images, {len(gt)} poses")

        baseline = Path(task["baseline_dir"])
        os.chdir(baseline)
        sys.path.insert(0, str(baseline))
        native = importlib.import_module("vggt_long")
        config = native.load_config(task["baseline_config"])
        config["Model"]["loop_enable"] = False
        if config["Model"].get("calib", False):
            raise ValueError("Model.calib requires extra calibration; this runner supplies images only")
        if task.get("checkpoint"):
            key = {"VGGT": "VGGT", "Pi3": "Pi3", "Mapanything": "Map"}[config["Weights"]["model"]]
            config["Weights"][key] = task["checkpoint"]

        workspace = Path(task["workspace"])
        staged = workspace / "images"
        staged.mkdir()
        for index, path in enumerate(images):
            suffix = ".png" if path.suffix.lower() == ".png" else ".jpg"
            (staged / f"{index:09d}{suffix}").symlink_to(path.resolve())

        if config["Model"]["align_method"] == "numba":
            native.warmup_numba()
        pipeline = None
        try:
            with pose_only_exports(native):
                pipeline = native.VGGT_Long(str(staged), str(workspace / "native"), config)
                pipeline.run()
            record = evaluate_scene(workspace / "native" / "camera_poses.txt",
                                    task["gt_path"], output, scene)
            record["status"] = "ok"
        finally:
            if pipeline is not None:
                pipeline.close()
    except Exception as exc:
        traceback.print_exc()
        record = {"status": "error", "error": str(exc)}
    with Path(task["result_path"]).open("w") as stream:
        json.dump(record, stream, indent=2, allow_nan=False)
    return 0 if record["status"] == "ok" else 1


def run_scene(scene, dataset_output, settings, device):
    log_path = dataset_output / f"{scene['scene']}.log"
    settings["scratch_root"].mkdir(parents=True, exist_ok=True)
    with tempfile.TemporaryDirectory(prefix="scene-", dir=settings["scratch_root"]) as temporary:
        workspace = Path(temporary)
        result_path = workspace / "result.json"
        task_path = workspace / "task.json"
        task = {**scene, "output_dir": str(dataset_output), "workspace": temporary,
                "result_path": str(result_path),
                "baseline_dir": str(settings["baseline_dir"]),
                "baseline_config": str(settings["baseline_config"]),
                "checkpoint": str(settings["checkpoint"]) if settings["checkpoint"] else None}
        task_path.write_text(json.dumps(task))
        env = os.environ.copy()
        env.update(CUDA_VISIBLE_DEVICES=str(device), PYTHONUNBUFFERED="1",
                   PYTHONDONTWRITEBYTECODE="1")
        env.setdefault("NUMBA_CACHE_DIR", str(settings["scratch_root"] / "numba-cache"))
        process = None
        try:
            with log_path.open("w") as log:
                with _PROCESS_LOCK:
                    if _CANCEL.is_set():
                        return {"status": "error", "error": "Interrupted", "log": str(log_path)}
                    process = subprocess.Popen(
                        [settings["python_executable"], str(Path(__file__).resolve()),
                         "--_task", str(task_path)], cwd=settings["baseline_dir"],
                        env=env, stdout=log, stderr=subprocess.STDOUT, start_new_session=True)
                    _PROCESSES.add(process)
                returncode = process.wait()
            if result_path.is_file():
                record = json.loads(result_path.read_text())
                if returncode and record.get("status") == "ok":
                    record = {"status": "error", "error": f"Worker exited with code {returncode}"}
            else:
                record = {"status": "error", "error": f"Worker exited with code {returncode} without a result"}
        except Exception as exc:
            with log_path.open("a") as log:
                traceback.print_exc(file=log)
            record = {"status": "error", "error": str(exc)}
        finally:
            if process is not None:
                with _PROCESS_LOCK:
                    _PROCESSES.discard(process)
        record["log"] = str(log_path)
        return record


def stop_workers():
    _CANCEL.set()
    with _PROCESS_LOCK:
        processes = list(_PROCESSES)
    for process in processes:
        if process.poll() is None:
            try:
                os.killpg(process.pid, signal.SIGTERM)
            except ProcessLookupError:
                pass
    for process in processes:
        try:
            process.wait(timeout=5)
        except subprocess.TimeoutExpired:
            try:
                os.killpg(process.pid, signal.SIGKILL)
            except ProcessLookupError:
                pass
            process.wait()


def run_dataset(name, prepared_root, settings, devices):
    output = settings["output_root"] / name
    for group in ("before_alignment", "after_alignment"):
        (output / group).mkdir(parents=True, exist_ok=True)
    scenes, skipped = discover_scenes(prepared_root, settings["image_subdir"])
    records = {}

    def worker(device, assigned):
        result = []
        for scene in assigned:
            if _CANCEL.is_set():
                break
            result.append((scene["scene"], run_scene(scene, output, settings, device)))
        return result

    executor = ThreadPoolExecutor(max_workers=len(devices))
    try:
        futures = [executor.submit(worker, device, assigned)
                   for device, assigned in split_scenes(scenes, devices) if assigned]
        for future in as_completed(futures):
            records.update(future.result())
    except KeyboardInterrupt:
        stop_workers()
        raise
    finally:
        executor.shutdown(wait=True, cancel_futures=True)

    records = dict(sorted(records.items()))
    failed = [scene for scene, record in records.items() if record.get("status") == "error"]
    summary = {"dataset": name, "scenes": records, **aggregate_scene_metrics(records),
               "skipped_scenes": skipped, "failed_scenes": failed}
    with (output / "metrics.json").open("w") as stream:
        json.dump(summary, stream, indent=2, allow_nan=False)
    if failed:
        sys.stderr.write(f"\nError: {name}: {len(failed)} scene(s) failed; log: {output / (failed[0] + '.log')}\n")
    return not failed


def check_runtime(settings, devices):
    if settings["checkpoint"] and not settings["checkpoint"].exists():
        raise FileNotFoundError(f"Checkpoint not found: {settings['checkpoint']}")
    env = os.environ.copy()
    env["CUDA_VISIBLE_DEVICES"] = ",".join(devices)
    probe = subprocess.run(
        [settings["python_executable"], "-c",
         "import torch,json; print(json.dumps({'cuda':torch.cuda.is_available(),'count':torch.cuda.device_count()}))"],
        env=env, capture_output=True, text=True)
    if probe.returncode:
        raise RuntimeError(probe.stderr.strip() or "Cannot load the baseline's PyTorch environment")
    state = json.loads(probe.stdout)
    if not state["cuda"] or state["count"] != len(devices):
        raise RuntimeError(f"CUDA cannot access all selected GPUs ({','.join(devices)})")


def main(argv=None):
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--config", type=Path, default=PROJECT_ROOT / "configs/vo-vggt-long.yaml")
    parser.add_argument("--datasets", nargs="+", default=["all"], help="all, or configured dataset names")
    parser.add_argument("--gpus", default="0,1,2,3", help="GPU indices within the current visibility mask")
    parser.add_argument("--_task", type=Path, help=argparse.SUPPRESS)
    args = parser.parse_args(argv)
    if args._task:
        return scene_worker(args._task)
    settings = None
    shown = False
    try:
        settings = load_settings(args.config)
        devices = resolve_devices(args.gpus, os.environ.get("CUDA_VISIBLE_DEVICES"))
        names = list(settings["datasets"]) if args.datasets == ["all"] else args.datasets
        if len(set(names)) != len(names) or any(name not in settings["datasets"] for name in names):
            raise ValueError(f"Select dataset names from: {', '.join(settings['datasets'])}, or all")
        check_runtime(settings, devices)
        success = True
        width = max(map(len, names), default=0)
        for name in names:
            print(f"\rTesting dataset: {name:<{width}}", end="", flush=True)
            shown = True
            success = run_dataset(name, settings["datasets"][name], settings, devices) and success
        print()
        return 0 if success else 1
    except KeyboardInterrupt:
        stop_workers()
        sys.stderr.write("\nInterrupted.\n")
        return 130
    except Exception as exc:
        root = settings["output_root"] if settings else PROJECT_ROOT / "results/vo-vggt-long"
        root.mkdir(parents=True, exist_ok=True)
        log_path = root / "runner.log"
        with log_path.open("w") as log:
            traceback.print_exc(file=log)
        prefix = "\n" if shown else ""
        sys.stderr.write(f"{prefix}Error: {str(exc).splitlines()[0]}; log: {log_path}\n")
        return 1


if __name__ == "__main__":
    raise SystemExit(main())
