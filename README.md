# VO Baselines

Baseline repositories are Git submodules. Shared adapters, configuration, and
evaluation live alongside them.

## Reusable adapter prompt

Replace `<BASELINE_DIR>` with the baseline folder name, such as `vo-vggt-long`.

```text
Create a visual-odometry runner for this baseline:

Project: /home/lei/workspace/vo_baselines
Baseline: /home/lei/workspace/vo_baselines/<BASELINE_DIR>
Script: /home/lei/workspace/vo_baselines/adapters/<BASELINE_DIR>/run.py
Datasets: /data/lei/vo_benchmarks
Evaluator: /home/lei/workspace/vo_baselines/eval

1. Reuse the baseline's existing test/inference code through an extra adapter
   script. Do not change its core model, preprocessing, or pose-estimation logic.
   Briefly explain whether an extra script is sufficient. If original test/I/O
   code needs changes, explain which files and why before making those changes.

2. Load the already-prepared images from sequences/<scene>/image_2/ and GT from
   poses/<scene>.txt. Find the prepared roots under the dataset collection; for
   example, euroc/undistorted. Process every scene with matching nonempty GT.
   Skip missing/empty GT. Keep scene IDs, sorted image order, and corresponding
   GT rows consistent. Do not inspect dataset manifests, conversion scripts,
   calibration, frame mappings, or validity/segment metadata, or reprocess data.
   If a baseline requires an extra input, state specifically what is missing.

3. Read how the baseline represents and exports poses, then convert its output
   to match the prepared GT and existing evaluator: relative/absolute poses,
   C2W/W2C direction, camera frame/axes, units, and image correspondence.
   EuRoC's prepared GT uses 12 row-major values per 3x4 camera-to-first-camera
   pose, in meters, with the first pose identity. Preserve native prediction
   scale; any GT-fitted alignment is evaluation-only. Reuse the existing eval
   module without changing its metric definitions.

4. Disable loop closure, global bundle adjustment, and optional final/global
   trajectory optimization through the baseline's existing controls. Keep local
   optimization required by its original odometry estimator. If disabling a
   backend requires changing core code, explain the limitation instead.

5. Inspect what the baseline saves and disable optional pointmap, depth,
   point-cloud, mesh, and other non-pose exports to avoid unnecessary disk usage.
   Preserve computations and temporary intermediates required to estimate poses;
   suppress optional saving rather than changing the estimator.

6. Run datasets sequentially in the configured order. For each dataset, distribute
   its eligible scenes roughly equally across the selected GPUs: scene counts
   differ by at most one. Use one worker per GPU and process each scene once.
   Wait for all workers on the current dataset before starting the next dataset.
   Support --config, --datasets all, and --gpus 0,1,2,3.

7. Compute per-scene metrics and per-dataset averages separately for
   before_alignment and after_alignment. Average each metric over its own
   valid scene values, excluding N/A from its numerator and denominator.
   Keep valid zeros. N/A translation/rotation errors must not exclude a scene's
   valid ATE/ASE. Report contributor counts; if none are valid, report N/A.

8. Keep terminal output to one status line: Testing dataset: <dataset-name>.
   Update the same line when moving to the next dataset. Save verbose baseline
   output, per-scene details, and metrics to result/log files instead of
   printing them. If execution fails, show only a brief error and the log path.

Save any needed configuration in configs/<BASELINE_DIR>.yaml. Save all results
under /home/lei/workspace/vo_baselines/results/<BASELINE_DIR>/<dataset-name>/
using this layout, without per-scene folders:
  before_alignment/<scene>.txt
  after_alignment/<scene>.txt
  metrics.json
  <scene>.log
Use scene IDs as pose filenames, for example 00.txt and 01.txt.
before_alignment contains poses in the evaluator's format/frame with native
scale. after_alignment contains poses returned by the existing evaluator's
GT-fitted 7dof alignment. If alignment is unavailable, report N/A instead of
writing a fabricated aligned trajectory.
Use one metrics.json for all scenes' before/after metrics, dataset averages,
and contributor counts. Save verbose scene logs directly as <scene>.log.
Give the command for running all datasets on four GPUs.

Keep the work limited to these requirements. Do not add Git/fork/diff/provenance
audits, dataset-metadata audits, or unrelated features.
```

## Expected command after the adapter is implemented

```bash
cd /home/lei/workspace/vo_baselines
python adapters/<BASELINE_DIR>/run.py \
    --config configs/<BASELINE_DIR>.yaml \
    --datasets all \
    --gpus 0,1,2,3
```
