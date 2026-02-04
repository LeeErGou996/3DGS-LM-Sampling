# 3DGS-LM Sampling Modules Documentation

This document outlines the code structure related to **LHS (Left-Hand Side) subsampling** in the 3DGS-LM project. The sampling logic has been modularized to separate concerns between sampling strategies, visualization, numerical solving, and the training loop.

## I. Module Overview

| Module | Responsibility | Main Interfaces |
|:---|:---|:---|
| **`fps_utils.py`** | FPS Sampling & Voronoi Weighting | `get_lhs_indices_fps_vectorized`, `compute_lhs_weights_voronoi` |
| **`camera_vis.py`** | Visualization Tools | `plot_camera_frustum_geometry_corrected`, `visualize_coverage_comparison` |
| **`ssgn_core.py`** | SSGN Numerical Solver Core | `linear_solve_ssgn`, `lm_step_ssgn` (Integration of sampling & solving) |
| **`ssgn_training.py`** | Training Loop Logic | `training_ssgn`, `training_report`, `log_render_stats` |

**Entry Point:** `train.py` invokes the complete training process via `from ssgn_training import training_ssgn`.

---

## II. Detailed Descriptions

### 1. `fps_utils.py` — FPS & Voronoi

* **`get_lhs_indices_fps_vectorized`**
    * Performs **Farthest Point Sampling (FPS)** on the RHS subset or the full set.
    * **Goal:** Ensures the LHS set is spatially distributed evenly to maximize geometric coverage, stabilizing the Hessian estimation.

* **`compute_lhs_weights_voronoi`**
    * Calculates compensation weights for each LHS camera based on **Voronoi regions**.
    * **Mechanism:** Each camera in the RHS set "votes" for its nearest LHS neighbor.

### 2. `camera_vis.py` — Visualization

* **`visualize_coverage_comparison`**
    * Generates 3D scatter plots showing the spatial distribution of "Selected (LHS)" vs. "Unselected" cameras (typically Red vs. Blue).
    * Used to verify the effectiveness of the sampling strategy.

### 3. `ssgn_core.py` — SSGN Core

* **`linear_solve_ssgn`**
    * The decoupled linear solver. It accepts two different inputs: high-resolution RHS (for gradient $g$) and low-resolution/subsampled LHS (for Hessian approximation $H$).

* **`lm_step_ssgn`**
    * Wrapper for a single Levenberg-Marquardt step.
    * **Execution flow:** `FPS Sampling` -> `Voronoi Weighting` -> `Vis (Optional)` -> `Linear Solve` -> `Update`.

### 4. `ssgn_training.py` — Training Logic

* **`training_ssgn`**
    * Replaces the original training function. Manages checkpoints, SGD warmup, and the main LM loop.

---

## III. Call Relationship Diagram

```text
train.py 
  └── calls ssgn_training.py
       └── calls ssgn_core.py (lm_step_ssgn)
            ├── 1. Sampling: fps_utils.py (get_lhs_indices_fps)
            ├── 2. Weighting: fps_utils.py (compute_lhs_weights_voronoi)
            ├── 3. Plotting: camera_vis.py (visualize_coverage)
            └── 4. Solving: ssgn_core.py (linear_solve_ssgn)
## IV. How to Run
Prerequisites: Ensure diff-gaussian-rasterization and simple-knn are installed.

### Example 1: Full SSGN Training (Recommended)
Enables FPS sampling, Voronoi weighting, and SSGN mode.

Bash
python train.py \
    -s /path/to/dataset/garden \
    --root_out ./output/garden_fps \
    --eval \
    --images images_4 \
    --enable_ssgn \
    --enable_fps_lhs \
    --use_voronoi_weights \
    --ssgn_size_rhs 20 \
    --ssgn_size_lhs 12 \
    --num_sgd_iterations_before_gn 500
    
### Example 2: Low Memory Mode (Consumer GPU)
Uses LHS Downsampling (--lhs_downsample_scale) to significantly reduce VRAM usage. For example, downscaling LHS images by 4x for Hessian computation, while keeping RHS at full resolution for gradients.

Bash
python train.py \
    -s /path/to/dataset/garden \
    --root_out ./output/garden_low_mem \
    --eval \
    --enable_ssgn \
    --enable_fps_lhs \
    --lhs_downsample_scale 4.0 \
    --ssgn_size_lhs 10 \
    --iterations 3000