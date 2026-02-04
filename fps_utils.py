from typing import List

import torch


def get_lhs_indices_fps_vectorized(viewpoint_stack, rhs_indices, n_lhs, device: str = "cuda") -> List[int]:
    """
    Vectorized farthest point sampling (FPS) to select a LHS subset from RHS.
    This ensures the LHS cameras are as spatially spread out as possible,
    which stabilizes curvature estimation.
    """
    n_rhs = len(rhs_indices)
    if n_rhs <= n_lhs:
        return list(rhs_indices)

    rhs_centers = torch.stack([viewpoint_stack[i].camera_center for i in rhs_indices]).to(device)

    selected_local_indices = [0]
    min_dists_sq = torch.sum((rhs_centers - rhs_centers[0]) ** 2, dim=1)
    min_dists_sq[0] = -1.0

    for _ in range(n_lhs - 1):
        farthest_idx = torch.argmax(min_dists_sq).item()
        selected_local_indices.append(farthest_idx)

        new_pivot = rhs_centers[farthest_idx]
        dists_to_pivot_sq = torch.sum((rhs_centers - new_pivot) ** 2, dim=1)
        min_dists_sq = torch.min(min_dists_sq, dists_to_pivot_sq)
        min_dists_sq[farthest_idx] = -1.0

    final_lhs_indices = [rhs_indices[i] for i in selected_local_indices]
    return final_lhs_indices


def compute_lhs_weights_voronoi(viewpoint_stack, rhs_indices, lhs_indices, device: str = "cuda") -> torch.Tensor:
    """
    Compute Voronoi-style weights for each LHS camera:
    each RHS camera votes for its closest LHS camera.
    """
    rhs_centers = torch.stack([viewpoint_stack[i].camera_center for i in rhs_indices]).to(device)
    lhs_centers = torch.stack([viewpoint_stack[i].camera_center for i in lhs_indices]).to(device)

    r_sq = torch.sum(rhs_centers ** 2, dim=1, keepdim=True)
    l_sq = torch.sum(lhs_centers ** 2, dim=1, keepdim=True)
    dist_sq = r_sq + l_sq.t() - 2 * torch.mm(rhs_centers, lhs_centers.t())

    nearest_lhs_idx = torch.argmin(dist_sq, dim=1)
    weights = torch.bincount(nearest_lhs_idx, minlength=len(lhs_indices)).float()
    return weights

