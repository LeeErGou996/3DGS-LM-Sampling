import os
from typing import List

import numpy as np
import torch
import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt  # noqa: E402
from mpl_toolkits.mplot3d import Axes3D  # noqa: F401,E402
from mpl_toolkits.mplot3d.art3d import Poly3DCollection  # noqa: E402


def plot_camera_frustum_geometry_corrected(ax, camera, color: str = "r", scale: float = 1.0, alpha: float = 0.5):
    """
    Plot a single camera frustum (wireframe + semi-transparent plane) in world space.
    """
    if isinstance(camera.R, torch.Tensor):
        R_w2c = camera.R.detach().cpu().numpy()
        T_w2c = camera.T.detach().cpu().numpy()
    else:
        R_w2c = camera.R
        T_w2c = camera.T

    W2C = np.eye(4)
    W2C[:3, :3] = R_w2c
    W2C[:3, 3] = T_w2c

    try:
        C2W = np.linalg.inv(W2C)
    except np.linalg.LinAlgError:
        return

    cam_pos = C2W[:3, 3]
    right = C2W[:3, 0]
    down = C2W[:3, 1]
    forward = C2W[:3, 2]

    to_origin = -cam_pos
    dot = np.dot(forward, to_origin)
    if dot < 0:
        forward = -forward
        right = -right

    aspect_ratio = 1.5
    if hasattr(camera, "image_width") and hasattr(camera, "image_height"):
        aspect_ratio = camera.image_width / camera.image_height

    d_phys = scale
    h_phys = d_phys * 0.7
    w_phys = h_phys * aspect_ratio

    plane_center = cam_pos + forward * d_phys

    tl = plane_center - w_phys * right - h_phys * down
    tr = plane_center + w_phys * right - h_phys * down
    br = plane_center + w_phys * right + h_phys * down
    bl = plane_center - w_phys * right + h_phys * down

    corners = np.array([tl, tr, br, bl])

    for corner in corners:
        ax.plot(
            [cam_pos[0], corner[0]],
            [cam_pos[1], corner[1]],
            [cam_pos[2], corner[2]],
            color=color,
            linewidth=0.8,
            alpha=alpha + 0.2,
        )

    border_x = [tl[0], tr[0], br[0], bl[0], tl[0]]
    border_y = [tl[1], tr[1], br[1], bl[1], tl[1]]
    border_z = [tl[2], tr[2], br[2], bl[2], tl[2]]
    ax.plot(border_x, border_y, border_z, color=color, linewidth=1.0, alpha=alpha + 0.3)

    verts = [list(zip(border_x[:-1], border_y[:-1], border_z[:-1]))]
    poly = Poly3DCollection(verts, facecolors=color, alpha=alpha * 0.4)
    ax.add_collection3d(poly)

    ax.scatter(cam_pos[0], cam_pos[1], cam_pos[2], color=color, s=15, marker="o", alpha=0.8)


def visualize_coverage_comparison(viewpoint_stack: List, selected_indices, title: str = None, save_path: str = None):
    """
    Visualize coverage:
    red = selected (e.g. LHS),
    blue = unselected (rest).
    """
    total_num = len(viewpoint_stack)
    selected_set = set(selected_indices)
    unselected_indices = [i for i in range(total_num) if i not in selected_set]

    print(
        f"[Visualizer] Coverage Check: {len(selected_indices)} Selected (Red) vs {len(unselected_indices)} Unselected (Blue)"
    )

    centers_list = []
    for cam in viewpoint_stack:
        if isinstance(cam.camera_center, torch.Tensor):
            centers_list.append(cam.camera_center.detach().cpu().numpy())
        else:
            centers_list.append(np.array(cam.camera_center))
    all_centers = np.stack(centers_list)

    fig = plt.figure(figsize=(14, 10))
    ax = fig.add_subplot(111, projection="3d")
    if title:
        ax.set_title(title, fontsize=14)
    else:
        ax.set_title(
            f"Coverage: Red={len(selected_indices)} (FPS), Blue={len(unselected_indices)} (Others)",
            fontsize=14,
        )

    scene_radius = np.linalg.norm(all_centers.max(0) - all_centers.min(0)) * 0.5
    if scene_radius == 0:
        scene_radius = 1.0
    frustum_scale = scene_radius * 0.15

    for idx in unselected_indices:
        cam = viewpoint_stack[idx]
        plot_camera_frustum_geometry_corrected(ax, cam, color="royalblue", scale=frustum_scale, alpha=0.2)

    for idx in selected_indices:
        cam = viewpoint_stack[idx]
        plot_camera_frustum_geometry_corrected(ax, cam, color="red", scale=frustum_scale * 1.05, alpha=0.7)

    from matplotlib.lines import Line2D

    legend_elements = [
        Line2D([0], [0], color="red", lw=2, label="Selected (LHS)"),
        Line2D([0], [0], color="royalblue", lw=2, label="Unselected (Rest)"),
    ]
    ax.legend(handles=legend_elements, loc="upper right")

    ax.set_xlabel("X")
    ax.set_ylabel("Y")
    ax.set_zlabel("Z")

    x_limits = [all_centers[:, 0].min(), all_centers[:, 0].max()]
    y_limits = [all_centers[:, 1].min(), all_centers[:, 1].max()]
    z_limits = [all_centers[:, 2].min(), all_centers[:, 2].max()]

    x_range = x_limits[1] - x_limits[0]
    y_range = y_limits[1] - y_limits[0]
    z_range = z_limits[1] - z_limits[0]
    max_range = max(x_range, y_range, z_range)
    mid_x, mid_y, mid_z = np.mean(x_limits), np.mean(y_limits), np.mean(z_limits)

    ax.set_xlim(mid_x - max_range / 2, mid_x + max_range / 2)
    ax.set_ylim(mid_y - max_range / 2, mid_y + max_range / 2)
    ax.set_zlim(mid_z - max_range / 2, mid_z + max_range / 2)

    if save_path:
        os.makedirs(os.path.dirname(save_path), exist_ok=True)
        try:
            plt.savefig(save_path, dpi=120, bbox_inches="tight")
            print(f"[Visualizer] Saved coverage comparison to {save_path}")
        except Exception as e:
            print(f"[Visualizer] Error saving plot: {e}")
        finally:
            plt.close(fig)
    else:
        plt.show()


# Backwards-compatible alias with the name used in lm_step_ssgn
visualize_cameras_with_projection = visualize_coverage_comparison

