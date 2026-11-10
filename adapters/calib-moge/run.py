"""Estimate MoGe-2 metric depth and camera intrinsics for the prepared VO datasets.

MoGe's own inference code is used unchanged. For every scene with frames in
<prepared_root>/sequences/<scene>/image_2/, this writes

  <prepared_root>/moge2/depth/<scene>/<frame>.png  uint16 PNG depth in centimetres, 0 = invalid
  <prepared_root>/moge2/intrinsics/<scene>.txt     one "fx fy cx cy" row per sorted frame

The depth encoding matches the existing Metric3D depth maps, and each intrinsics
file matches the wildcam_K/<scene>.txt layout (tab-separated, %.9f, pixel units).
"""

from __future__ import annotations

import argparse
from collections import deque
from concurrent.futures import ThreadPoolExecutor
import json
import os
from pathlib import Path
import signal
import subprocess
import sys
import tempfile
import threading
import time
import traceback


PROJECT_ROOT = Path(__file__).resolve().parents[2]
IMAGE_SUFFIXES = {".png", ".jpg", ".jpeg"}
DEPTH_SCALE = 100.0  # store depth in centimetres, like the Metric3D depth maps
DEFAULT_CHECKPOINT = "Ruicheng/moge-2-vitl-normal"  # native `moge infer --version v2` default
PREFETCH = 4  # frames decoded ahead of the GPU
MAX_PENDING_WRITES = 8  # depth maps still being encoded behind the GPU
_CANCEL = threading.Event()
_PROCESS_LOCK = threading.Lock()
_PROCESSES = set()


def list_images(image_dir):
    """Sorted JPG/JPEG/PNG frames; this order defines the intrinsics rows."""
    with os.scandir(image_dir) as entries:
        return sorted(Path(entry.path) for entry in entries
                      if entry.is_file() and Path(entry.name).suffix.lower() in IMAGE_SUFFIXES)


def discover_scenes(prepared_root, settings):
    prepared_root = Path(prepared_root)
    sequences = prepared_root / "sequences"
    if not sequences.is_dir():
        raise FileNotFoundError(f"Expected sequences/ under {prepared_root}")
    output = prepared_root / settings["output_subdir"]
    scenes, without_images = [], []
    for directory in sorted(path for path in sequences.iterdir() if path.is_dir()):
        image_dir = directory / settings["image_subdir"]
        frames = len(list_images(image_dir)) if image_dir.is_dir() else 0
        if not frames:
            without_images.append(directory.name)
            continue
        scenes.append({
            "scene": directory.name,
            "image_dir": str(image_dir),
            "frames": frames,
            "depth_dir": str(output / "depth" / directory.name),
            "intrinsics_path": str(output / "intrinsics" / f"{directory.name}.txt"),
        })
    return scenes, without_images


def scene_complete(scene):
    """A scene is done when its intrinsics file (written last) and all depth maps exist."""
    path = Path(scene["intrinsics_path"])
    if not path.is_file():
        return False
    with path.open() as stream:
        rows = [line.split() for line in stream if line.strip()]
    if len(rows) != scene["frames"] or any(len(row) != 4 for row in rows):
        return False
    depth_dir = Path(scene["depth_dir"])
    return all((depth_dir / f"{image.stem}.png").is_file()
               for image in list_images(scene["image_dir"]))


def split_scenes(scenes, devices):
    """Deal scenes largest-first in serpentine rounds, one scene per GPU per round.

    Per-GPU scene counts therefore differ by at most one, while alternating the
    dealing direction keeps the number of frames per GPU close.
    """
    if not devices:
        raise ValueError("Select at least one GPU")
    groups = [[] for _ in devices]
    ordered = sorted(scenes, key=lambda scene: (-scene["frames"], scene["scene"]))
    for index, scene in enumerate(ordered):
        round_index, slot = divmod(index, len(devices))
        groups[slot if round_index % 2 == 0 else len(devices) - 1 - slot].append(scene)
    return [(device, sorted(group, key=lambda scene: scene["scene"]))
            for device, group in zip(devices, groups)]


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
    checkpoint = Path(str(config.get("checkpoint") or DEFAULT_CHECKPOINT)).expanduser()
    if checkpoint.is_absolute() or checkpoint.suffix in {".pt", ".pth"}:
        # Local file, kept as configured (not symlink-resolved into the HF blob store).
        checkpoint = checkpoint if checkpoint.is_absolute() else PROJECT_ROOT / checkpoint
    checkpoint = str(checkpoint)  # anything else is a Hugging Face repo ID
    resolution_level = int(config.get("resolution_level", 9))
    if not 0 <= resolution_level <= 9:
        raise ValueError("resolution_level must be an integer from 0 to 9")
    return {
        "baseline_dir": absolute(config.get("baseline_dir", "calib-moge")),
        "checkpoint": checkpoint,
        "python_executable": str(config.get("python_executable") or sys.executable),
        "image_subdir": config.get("image_subdir", "image_2"),
        "output_subdir": config.get("output_subdir", "moge2"),
        "log_root": absolute(config.get("log_root", "results/calib-moge")),
        "resolution_level": resolution_level,
        "fp16": bool(config.get("fp16", False)),
        "datasets": {name: (Path(root) if Path(root).is_absolute() else data_root / root)
                     for name, root in config["datasets"].items()},
    }


# ------------------------------------------------------------------ GPU worker


def load_image(path):
    """Native `moge infer` preprocessing: OpenCV color decode, BGR to RGB, scale to [0, 1]."""
    import cv2
    import numpy as np

    image = cv2.imread(str(path))
    if image is None:
        raise ValueError(f"Could not read image: {path}")
    image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
    return (image / 255).astype(np.float32)


def save_depth(path, depth):
    """Metric3D depth-map encoding: uint16 centimetres; nonfinite (masked) pixels become 0."""
    import cv2
    import numpy as np

    depth_png = np.nan_to_num(depth, nan=0.0, posinf=0.0, neginf=0.0)
    depth_png = np.clip(
        np.rint(depth_png * DEPTH_SCALE), 0, np.iinfo(np.uint16).max
    ).astype(np.uint16)
    if not cv2.imwrite(str(path), depth_png):
        raise OSError(f"Could not write depth map: {path}")


def pixel_intrinsics(normalized, width, height):
    """MoGe's normalized UV intrinsics to OpenCV pixel-center [fx, fy, cx, cy].

    K_pixels = [[W, 0, -0.5], [0, H, -0.5], [0, 0, 1]] @ K_normalized
    """
    return (normalized[0, 0] * width, normalized[1, 1] * height,
            normalized[0, 2] * width - 0.5, normalized[1, 2] * height - 0.5)


def process_scene(model, scene, task, readers, writers, emit):
    import numpy as np
    import torch

    images = list_images(scene["image_dir"])
    if len(images) != scene["frames"]:
        raise RuntimeError(f"Found {len(images)} images; expected {scene['frames']}")
    if len({image.stem for image in images}) != len(images):
        raise ValueError("Image names share a stem, so their depth maps would collide")
    depth_dir = Path(scene["depth_dir"])
    intrinsics_path = Path(scene["intrinsics_path"])
    depth_dir.mkdir(parents=True, exist_ok=True)
    intrinsics_path.parent.mkdir(parents=True, exist_ok=True)
    # The intrinsics file is written last and marks a finished scene, so drop a
    # stale one before any depth map is overwritten.
    intrinsics_path.unlink(missing_ok=True)

    reads = deque(readers.submit(load_image, image) for image in images[:PREFETCH])
    writes = deque()
    rows = []
    for index, image_path in enumerate(images):
        image = reads.popleft().result()
        if index + PREFETCH < len(images):
            reads.append(readers.submit(load_image, images[index + PREFETCH]))
        height, width = image.shape[:2]
        output = model.infer(
            torch.from_numpy(image).to(model.device).permute(2, 0, 1),
            fov_x=None,  # estimate the camera; no FoV prior
            resolution_level=task["resolution_level"],
            num_tokens=None,
            use_fp16=task["fp16"],
        )
        depth = output["depth"].cpu().numpy()
        intrinsics = pixel_intrinsics(
            output["intrinsics"].cpu().numpy().astype(np.float64), width, height)
        if depth.shape != (height, width):
            raise ValueError(f"{image_path.name}: depth shape {depth.shape} != image {(height, width)}")
        if not (np.isfinite(intrinsics).all() and min(intrinsics[:2]) > 0):
            raise ValueError(f"{image_path.name}: invalid intrinsics {intrinsics}")
        rows.append(intrinsics)
        writes.append(writers.submit(save_depth, depth_dir / f"{image_path.stem}.png", depth))
        while len(writes) > MAX_PENDING_WRITES:
            writes.popleft().result()
        emit(frames_done=1)
    while writes:
        writes.popleft().result()
    temporary = intrinsics_path.with_name(intrinsics_path.name + ".tmp")
    np.savetxt(temporary, np.asarray(rows, dtype=np.float64), fmt="%.9f", delimiter="\t")
    os.replace(temporary, intrinsics_path)
    return len(rows)


def worker_main(task_path):
    """Load MoGe-2 once on this worker's GPU, then process its scenes in order."""
    task = json.loads(Path(task_path).read_text())
    channel = os.fdopen(os.dup(1), "w", buffering=1)  # progress messages for the parent
    os.dup2(2, 1)  # anything else printed goes to this worker's log

    def emit(**message):
        channel.write(json.dumps(message) + "\n")

    baseline = Path(task["baseline_dir"]).resolve()
    # The moge2 env also has an editable install of another MoGe clone; use this submodule.
    sys.path.insert(0, str(baseline))
    import torch
    import moge
    from moge.model.v2 import MoGeModel

    if baseline not in Path(moge.__file__).resolve().parents:
        raise ImportError(f"moge was imported from {moge.__file__}, not {baseline}")
    model = MoGeModel.from_pretrained(task["checkpoint"]).to(torch.device("cuda")).eval()
    if task["fp16"]:
        model.half()  # same as `moge infer --fp16`
    print(f"Loaded MoGe-2 {task['checkpoint']} on GPU {task['device']} "
          f"(fp16={task['fp16']}, resolution_level={task['resolution_level']})", flush=True)

    with ThreadPoolExecutor(2) as readers, ThreadPoolExecutor(4) as writers:
        for scene in task["scenes"]:
            started = time.monotonic()
            print(f"[{scene['scene']}] {scene['frames']} frames from {scene['image_dir']}", flush=True)
            try:
                frames = process_scene(model, scene, task, readers, writers, emit)
            except Exception as exc:
                traceback.print_exc()
                emit(scene=scene["scene"], status="error", frames=scene["frames"],
                     error=f"{type(exc).__name__}: {exc}")
                torch.cuda.empty_cache()
                continue
            seconds = time.monotonic() - started
            print(f"[{scene['scene']}] done in {seconds:.1f} s ({frames / seconds:.2f} frames/s)",
                  flush=True)
            emit(scene=scene["scene"], status="ok", frames=frames, seconds=round(seconds, 1))
    channel.close()
    return 0


# ------------------------------------------------------------------ parent runner


def check_runtime(settings, devices):
    checkpoint = Path(settings["checkpoint"])
    if checkpoint.is_absolute() and not checkpoint.is_file():
        raise FileNotFoundError(f"Checkpoint not found: {checkpoint}")
    env = os.environ.copy()
    env.update(CUDA_VISIBLE_DEVICES=",".join(devices), PYTHONDONTWRITEBYTECODE="1")
    code = ("import json, sys; sys.path.insert(0, sys.argv[1]); import cv2, torch, moge; "
            "from moge.model.v2 import MoGeModel; "
            "print(json.dumps({'cuda': torch.cuda.is_available(), "
            "'count': torch.cuda.device_count(), 'moge': moge.__file__}))")
    probe = subprocess.run([settings["python_executable"], "-c", code, str(settings["baseline_dir"])],
                           cwd=settings["baseline_dir"], env=env, capture_output=True, text=True)
    if probe.returncode:
        detail = (probe.stderr.strip() or probe.stdout.strip() or "Unknown import error").splitlines()[-1]
        raise RuntimeError(f"MoGe environment is not ready: {detail}")
    state = json.loads(probe.stdout.splitlines()[-1])
    if settings["baseline_dir"] not in Path(state["moge"]).resolve().parents:
        raise RuntimeError(f"moge imports from {state['moge']}, not {settings['baseline_dir']}")
    if not state["cuda"] or state["count"] != len(devices):
        raise RuntimeError(f"CUDA cannot access all selected GPUs ({','.join(devices)})")


def launch_worker(device, scenes, settings, log_dir, workspace, worker_count, records, progress, lock):
    """Start one GPU worker and a thread that collects its progress and scene results."""
    log_path = log_dir / f"gpu{device}.log"
    task_path = workspace / f"gpu{device}.json"
    task_path.write_text(json.dumps({
        "device": device, "scenes": scenes,
        "baseline_dir": str(settings["baseline_dir"]), "checkpoint": settings["checkpoint"],
        "resolution_level": settings["resolution_level"], "fp16": settings["fp16"],
    }))
    env = os.environ.copy()
    env.update(CUDA_VISIBLE_DEVICES=str(device), PYTHONUNBUFFERED="1", PYTHONDONTWRITEBYTECODE="1")
    env.setdefault("OMP_NUM_THREADS", str(max(1, (os.cpu_count() or 1) // worker_count)))
    log = log_path.open("w")
    with _PROCESS_LOCK:
        if _CANCEL.is_set():
            log.close()
            raise KeyboardInterrupt
        process = subprocess.Popen(
            [settings["python_executable"], str(Path(__file__).resolve()), "--_worker", str(task_path)],
            cwd=settings["baseline_dir"], env=env, stdin=subprocess.DEVNULL,
            stdout=subprocess.PIPE, stderr=log, text=True, start_new_session=True)
        _PROCESSES.add(process)

    def watch():
        try:
            for line in process.stdout:
                try:
                    message = json.loads(line)
                except ValueError:
                    continue
                with lock:
                    if "frames_done" in message:
                        progress["done"] += message["frames_done"]
                    elif "scene" in message:
                        scene = message.pop("scene")
                        records[scene] = {**message, "gpu": device, "log": str(log_path)}
        finally:
            returncode = process.wait()
            log.close()
            with _PROCESS_LOCK:
                _PROCESSES.discard(process)
            with lock:
                for scene in scenes:
                    records.setdefault(scene["scene"], {
                        "status": "error", "frames": scene["frames"], "gpu": device,
                        "log": str(log_path),
                        "error": f"Worker exited with code {returncode} before finishing this scene"})

    thread = threading.Thread(target=watch, daemon=True)
    thread.start()
    return thread


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


_STATUS = {"text": None}


def show_status(name, width, progress):
    """Keep one terminal status line, redrawn in place only when it changes."""
    text = f"Processing dataset: {name:<{width}}  {progress['done']}/{progress['total']} frames"
    if text != _STATUS["text"]:
        _STATUS["text"] = text
        print(f"\r{text}" + ("\033[K" if sys.stdout.isatty() else ""), end="", flush=True)


def run_dataset(name, prepared_root, settings, devices, skip_existing, width):
    log_dir = settings["log_root"] / name
    log_dir.mkdir(parents=True, exist_ok=True)
    output = Path(prepared_root) / settings["output_subdir"]
    scenes, without_images = discover_scenes(prepared_root, settings)
    for subdir in ("depth", "intrinsics"):
        (output / subdir).mkdir(parents=True, exist_ok=True)
    finished = {scene["scene"] for scene in scenes if skip_existing and scene_complete(scene)}
    pending = [scene for scene in scenes if scene["scene"] not in finished]
    records = {scene["scene"]: {"status": "skipped", "frames": scene["frames"]}
               for scene in scenes if scene["scene"] in finished}
    progress = {"done": sum(record["frames"] for record in records.values()),
                "total": sum(scene["frames"] for scene in scenes)}
    lock = threading.Lock()
    threads = []
    with tempfile.TemporaryDirectory(prefix="calib-moge-") as workspace:
        try:
            for device, assigned in split_scenes(pending, devices):
                if assigned:
                    threads.append(launch_worker(device, assigned, settings, log_dir, Path(workspace),
                                                 len(devices), records, progress, lock))
            while any(thread.is_alive() for thread in threads):
                show_status(name, width, progress)
                time.sleep(0.5)
            show_status(name, width, progress)
        except KeyboardInterrupt:
            stop_workers()
            raise

    records = dict(sorted(records.items()))
    failed = [scene for scene, record in records.items() if record["status"] == "error"]
    summary = {
        "dataset": name,
        "prepared_root": str(prepared_root),
        "depth_dir": str(output / "depth"),
        "intrinsics_dir": str(output / "intrinsics"),
        "checkpoint": settings["checkpoint"],
        "resolution_level": settings["resolution_level"],
        "fp16": settings["fp16"],
        "scenes": records,
        "failed_scenes": failed,
        "scenes_without_images": without_images,
    }
    (log_dir / "summary.json").write_text(json.dumps(summary, indent=2, allow_nan=False) + "\n")
    if failed:
        sys.stderr.write(f"\nError: {name}: {len(failed)} scene(s) failed; "
                         f"log: {records[failed[0]]['log']}\n")
    return not failed


def main(argv=None):
    parser = argparse.ArgumentParser(description=__doc__,
                                     formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--config", type=Path, default=PROJECT_ROOT / "configs/calib-moge.yaml")
    parser.add_argument("--datasets", nargs="+", default=["all"], help="all, or configured dataset names")
    parser.add_argument("--gpus", default="0,1,2,3", help="GPU indices within the current visibility mask")
    parser.add_argument("--skip-existing", action="store_true",
                        help="skip scenes whose depth maps and intrinsics file are already complete")
    parser.add_argument("--_worker", type=Path, help=argparse.SUPPRESS)
    args = parser.parse_args(argv)
    if args._worker:
        return worker_main(args._worker)
    settings = None
    shown = False
    try:
        settings = load_settings(args.config)
        devices = resolve_devices(args.gpus, os.environ.get("CUDA_VISIBLE_DEVICES"))
        names = list(settings["datasets"]) if args.datasets == ["all"] else args.datasets
        if len(set(names)) != len(names) or any(name not in settings["datasets"] for name in names):
            raise ValueError(f"Select dataset names from: {', '.join(settings['datasets'])}, or all")
        check_runtime(settings, devices)
        width = max(map(len, names), default=0)
        success = True
        for name in names:
            shown = True
            success = run_dataset(name, settings["datasets"][name], settings, devices,
                                  args.skip_existing, width) and success
        print()
        return 0 if success else 1
    except KeyboardInterrupt:
        stop_workers()
        sys.stderr.write("\nInterrupted.\n")
        return 130
    except Exception as exc:
        root = settings["log_root"] if settings else PROJECT_ROOT / "results/calib-moge"
        root.mkdir(parents=True, exist_ok=True)
        log_path = root / "runner.log"
        with log_path.open("w") as log:
            traceback.print_exc(file=log)
        prefix = "\n" if shown else ""
        sys.stderr.write(f"{prefix}Error: {(str(exc) or type(exc).__name__).splitlines()[0]}; "
                         f"log: {log_path}\n")
        return 1


if __name__ == "__main__":
    raise SystemExit(main())
