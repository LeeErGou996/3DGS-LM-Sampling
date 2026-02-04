import math
import os
import random
from typing import Dict, List

import torch

from fps_utils import get_lhs_indices_fps_vectorized, compute_lhs_weights_voronoi
from camera_vis import visualize_cameras_with_projection
from train import (
    GaussianModel,
    OptimizationParams,
    PipelineParams,
    GSGNParams,
    GaussianRasterizer,
    render_all_images_and_backward,
    eval_jtf_and_get_sparse_jacobian,
    calc_preconditioner,
    apply_jtj,
    apply_j,
    cg_batch,
    render_and_add_to_residual_norm,
    get_residual_norm,
    measure_time,
)


# ==============================================================================
# 4. SSGN linear solver
# ==============================================================================
def linear_solve_ssgn(
    gaussians: GaussianModel,
    opt: OptimizationParams,
    pipe: PipelineParams,
    background: torch.Tensor,
    gsgn: GSGNParams,
    imgs_rhs: list,
    imgs_lhs: list,
    lhs_weights: torch.Tensor = None,
    lhs_downsample_scale: int = 1,
    trust_region_radius: float = 1.0,
    timing_dict: Dict[str, float] = None,
):
    with measure_time("ssgn_rhs_render", timing_dict):
        forward_output_rhs = render_all_images_and_backward(
            gaussians=gaussians,
            viewpoint_stack=imgs_rhs,
            opt=opt,
            pipe=pipe,
            background=background,
            compute_huber_weights=gsgn.compute_huber_weights,
            huber_c=gsgn.huber_c,
            compute_ssim_weights=gsgn.compute_ssim_weights,
            ssim_residual_scale_factor=gsgn.ssim_residual_scale_factor,
        )

    with measure_time("ssgn_rhs_eval", timing_dict):
        b, _, _, _, _ = eval_jtf_and_get_sparse_jacobian(
            gaussians=gaussians,
            forward_output=forward_output_rhs,
            timing_dict=timing_dict,
            use_double_precision=gsgn.use_double_precision,
        )

    del forward_output_rhs
    torch.cuda.empty_cache()

    with measure_time("ssgn_lhs_render", timing_dict):
        forward_output_lhs = render_all_images_and_backward(
            gaussians=gaussians,
            viewpoint_stack=imgs_lhs,
            opt=opt,
            pipe=pipe,
            background=background,
            compute_huber_weights=gsgn.compute_huber_weights,
            huber_c=gsgn.huber_c,
            compute_ssim_weights=gsgn.compute_ssim_weights,
            ssim_residual_scale_factor=gsgn.ssim_residual_scale_factor,
        )

    with measure_time("ssgn_lhs_eval", timing_dict):
        _, sparse_jacobians, index_maps, per_gaussian_caches, data = eval_jtf_and_get_sparse_jacobian(
            gaussians=gaussians,
            forward_output=forward_output_lhs,
            timing_dict=timing_dict,
            use_double_precision=gsgn.use_double_precision,
        )

    global_scale = len(imgs_rhs) / len(imgs_lhs)

    if lhs_weights is not None:
        try:
            weights_sqrt = torch.sqrt(lhs_weights).to(sparse_jacobians[0].device)
            for i in range(len(sparse_jacobians)):
                if i < len(weights_sqrt):
                    sparse_jacobians[i] *= weights_sqrt[i]
            scale_factor = 1.0
        except Exception as e:
            print(f"[WARNING] Failed to apply LHS weights: {e}. Falling back to global scaling.")
            scale_factor = global_scale
    else:
        scale_factor = global_scale

    segment_list = []
    segments_to_gaussians_list = []
    num_gaussians_in_block_list = []
    block_offset_in_segments_list = []

    with measure_time("sort_index_maps", timing_dict):
        for i in range(len(index_maps)):
            m = index_maps[i]
            half = m.numel() // 2
            gaussian_ids = m[:half]
            ray_ids = m[half:]
            sort_keys = gaussian_ids.to(torch.int64) * data.H * data.W * data.num_images + ray_ids.to(torch.int64)
            _, indices = torch.sort(sort_keys)
            sorted_gaussians = gaussian_ids[indices]
            index_maps[i] = ray_ids[indices]

            data_sub = GaussianRasterizer.subsample_data(data, [i])
            sparse_jacobians[i] = GaussianRasterizer.sort_sparse_jacobians(
                sparse_jacobians=[sparse_jacobians[i]],
                indices_map=[indices],
                data=data_sub,
                timing_dict=timing_dict,
            )[0]

            threads_per_block = 128
            num_blocks = math.ceil(len(sorted_gaussians) / threads_per_block)
            segments = torch.nonzero(sorted_gaussians[1:] - sorted_gaussians[:-1]).int().flatten() + 1
            block_borders = torch.arange(
                start=0,
                end=num_blocks,
                dtype=segments.dtype,
                device=segments.device,
            ) * threads_per_block
            segments = torch.cat([block_borders, segments])
            segments = torch.unique(segments, sorted=True)
            segments_to_block = segments // threads_per_block
            _, num_gaussians_in_block = torch.unique_consecutive(segments_to_block, return_counts=True)
            num_gaussians_in_block = num_gaussians_in_block.int()
            block_offset_in_segments = torch.cumsum(
                num_gaussians_in_block,
                dim=0,
                dtype=num_gaussians_in_block.dtype,
            )[:-1]
            block_offset_in_segments = torch.cat(
                [torch.zeros_like(block_offset_in_segments[0:1]), block_offset_in_segments]
            )

            segment_list.append(segments)
            segments_to_gaussians_list.append(sorted_gaussians[segments])
            num_gaussians_in_block_list.append(num_gaussians_in_block)
            block_offset_in_segments_list.append(block_offset_in_segments)

    with measure_time("build_M_CTC", timing_dict):
        M = calc_preconditioner(
            sparse_jacobians,
            index_maps,
            per_gaussian_caches,
            data,
            timing_dict,
            segment_list,
            segments_to_gaussians_list,
            num_gaussians_in_block_list,
            block_offset_in_segments_list,
        )

        M = M * scale_factor

        CTC = torch.clamp(M, gsgn.min_lm_diagonal, gsgn.max_lm_diagonal)
        M = 1.0 / (M + CTC)
        M = M.unsqueeze(-1)
        CTC = (1.0 / trust_region_radius) * CTC.unsqueeze(-1)

    def M_bmm(X):
        return (M * X[0]).unsqueeze(0)

    @torch.enable_grad()
    def A_bmm(X):
        x = X.squeeze()
        x_resorted = gaussians.get_resorted_vec(x)
        g = apply_jtj(
            x,
            x_resorted,
            sparse_jacobians,
            index_maps,
            per_gaussian_caches,
            data,
            timing_dict,
            segment_list,
            segments_to_gaussians_list,
            num_gaussians_in_block_list,
            block_offset_in_segments_list,
        )
        g = g * scale_factor

        x_resorted *= 0
        x_resorted += CTC.squeeze()
        x_resorted *= X.squeeze()
        g += x_resorted
        return g.unsqueeze(0).unsqueeze(-1)

    with measure_time("cg_batch", timing_dict):
        x, info = cg_batch(
            A_bmm,
            b.unsqueeze(0).unsqueeze(-1),
            M_bmm,
            maxiter=gsgn.pcg_max_iter,
            atol=gsgn.pcg_atol,
            rtol=gsgn.pcg_rtol,
            gradient_descent_every=gsgn.pcg_gradient_descent_every,
            explicit_residual_every=gsgn.pcg_explicit_residual_every,
            verbose=gsgn.pcg_verbose,
        )

    x = x.squeeze().float()
    val_max = x.abs().max()
    scale_factor_grad = min(1.0, gsgn.max_grad_norm / val_max)
    x = scale_factor_grad * x

    return {
        "x": x,
        "data": data,
        "forward_output": forward_output_lhs,
        "log_info": {
            "pcg_info": info,
            "mean_n_contrib_per_pixel": 0.0,
            "total_size_in_gb": 0.0,
            "sparse_jacobians_gb": 0.0,
            "index_maps_gb": 0.0,
            "per_gaussian_caches_gb": 0.0,
        },
    }


# ==============================================================================
# 5. LM Step with SSGN
# ==============================================================================
@torch.no_grad()
def lm_step_ssgn(
    gaussians,
    viewpoint_stack,
    viewpoint_stack_indices,
    opt,
    pipe,
    background,
    gsgn,
    iteration,
    trust_region_radius,
    radius_decrease_factor,
    timing_dict,
    forward_output,
    output_path=None,
    lhs_downsample_scale: int = 1,
):
    enable_ssgn = getattr(gsgn, "enable_ssgn", False)

    n_images = len(viewpoint_stack)

    n_rhs = getattr(gsgn, "ssgn_size_rhs", gsgn.image_subsample_size)
    if len(viewpoint_stack_indices) < n_rhs:
        viewpoint_stack_indices = list(range(n_images))
        random.shuffle(viewpoint_stack_indices)

    rhs_indices = viewpoint_stack_indices[:n_rhs]
    viewpoint_stack_indices = viewpoint_stack_indices[n_rhs:] + rhs_indices

    rhs_indices_sorted = sorted(rhs_indices)

    if enable_ssgn:
        n_lhs = getattr(gsgn, "ssgn_size_lhs", 10)
        enable_fps = getattr(gsgn, "enable_fps_lhs", False)

        if n_lhs >= n_rhs:
            if iteration == 1:
                print(f"[WARNING] SSGN enabled but LHS ({n_lhs}) >= RHS ({n_rhs}). Disabling SSGN (LHS=RHS).")
            enable_ssgn = False
            lhs_indices = rhs_indices_sorted
            frame_selection_method = "RHS"
        else:
            if enable_fps:
                lhs_indices = get_lhs_indices_fps_vectorized(viewpoint_stack, rhs_indices_sorted, n_lhs)
                frame_selection_method = "FPS"
            else:
                target_mode = getattr(gsgn, "image_subsample_frame_selection_mode", "strided")
                if target_mode == "random":
                    lhs_indices = random.sample(rhs_indices_sorted, n_lhs)
                    frame_selection_method = "random"
                else:
                    stride = max(1, len(rhs_indices_sorted) // n_lhs)
                    lhs_indices = rhs_indices_sorted[::stride][:n_lhs]
                    frame_selection_method = "strided"
    else:
        lhs_indices = rhs_indices_sorted
        frame_selection_method = "RHS"

    use_voronoi = getattr(gsgn, "use_voronoi_weights", False)

    lhs_weights = None
    if enable_ssgn and len(rhs_indices) > len(lhs_indices) and use_voronoi:
        try:
            lhs_weights = compute_lhs_weights_voronoi(viewpoint_stack, rhs_indices, lhs_indices)
        except Exception as e:
            print(f"[WARNING] Failed to compute Voronoi weights: {e}")
            lhs_weights = None

    if iteration == 1 or iteration % 10 == 0:
        try:
            if output_path is not None:
                os.makedirs(output_path, exist_ok=True)
                save_path = os.path.join(output_path, f"lhs_3d_projection_iter_{iteration}.png")
            else:
                save_path = f"lhs_3d_projection_iter_{iteration}.png"

            title = f"SSGN LHS Coverage - {frame_selection_method} (Iter {iteration})"
            visualize_cameras_with_projection(viewpoint_stack, lhs_indices, title=title, save_path=save_path)
        except Exception:
            pass

    imgs_rhs = [viewpoint_stack[i] for i in rhs_indices]
    imgs_lhs = [viewpoint_stack[i] for i in lhs_indices]

    imgs_lhs_for_solve = imgs_lhs
    if lhs_downsample_scale > 1:
        import torch.nn.functional as F
        from scene.cameras import Camera as CamCls

        if iteration == 1 or (iteration % 10 == 0):
            mode_info = "SSGN" if enable_ssgn else "Standard (LHS=RHS)"
            print(
                f"[{mode_info}] Iter {iteration}: Applying LHS downsampling scale={lhs_downsample_scale} "
                f"(Expected memory reduction: ~{100 * (1 - 1.0 / (lhs_downsample_scale ** 2)):.1f}%)"
            )

        imgs_lhs_for_solve = []
        ref_new_w, ref_new_h = None, None
        for cam in imgs_lhs:
            try:
                new_width = max(1, int(cam.image_width) // int(lhs_downsample_scale))
                new_height = max(1, int(cam.image_height) // int(lhs_downsample_scale))

                if ref_new_w is None:
                    ref_new_w, ref_new_h = new_width, new_height
                elif (new_width != ref_new_w) or (new_height != ref_new_h):
                    imgs_lhs_for_solve.append(cam)
                    continue

                orig_img = cam.original_image
                if orig_img is None:
                    imgs_lhs_for_solve.append(cam)
                    continue

                if not orig_img.is_cuda:
                    orig_img = orig_img.cuda()

                if orig_img.dim() != 3:
                    imgs_lhs_for_solve.append(cam)
                    continue

                downsampled_img = F.interpolate(
                    orig_img.unsqueeze(0),
                    size=(new_height, new_width),
                    mode="bilinear",
                    align_corners=False,
                    antialias=True,
                ).squeeze(0)

                W0 = float(cam.image_width)
                H0 = float(cam.image_height)
                sx = float(new_width) / max(1.0, W0)
                sy = float(new_height) / max(1.0, H0)

                new_fovx = float(cam.FoVx)
                new_fovy = float(cam.FoVy)

                new_cx = (float(cam.cx) * sx) if hasattr(cam, "cx") else float(new_width) * 0.5
                new_cy = (float(cam.cy) * sy) if hasattr(cam, "cy") else float(new_height) * 0.5

                if not (0.0 <= new_cx < float(new_width)) or not (0.0 <= new_cy < float(new_height)):
                    new_cx = float(new_width) * 0.5
                    new_cy = float(new_height) * 0.5

                cam_down = CamCls(
                    colmap_id=cam.colmap_id,
                    R=cam.R,
                    T=cam.T,
                    FoVx=new_fovx,
                    FoVy=new_fovy,
                    cx=new_cx,
                    cy=new_cy,
                    image=downsampled_img,
                    gt_alpha_mask=None,
                    image_name=f"{cam.image_name}_lhs_ds{lhs_downsample_scale}",
                    uid=cam.uid,
                    data_device=str(getattr(cam, "data_device", "cuda")),
                )

                imgs_lhs_for_solve.append(cam_down)
            except Exception as e:
                if iteration == 1:
                    print(f"[WARNING] Failed to downsample camera: {e}. Using original resolution.")
                imgs_lhs_for_solve.append(cam)

    if enable_ssgn:
        if iteration == 1 or iteration % 10 == 0:
            print(
                f"[SSGN] Iter {iteration}: RHS={len(imgs_rhs)}, LHS={len(imgs_lhs)}, "
                f"Mode={frame_selection_method}, Weighted={lhs_weights is not None}"
            )

    lhs_coverage_pct = None
    if enable_ssgn and len(rhs_indices) > 0 and len(lhs_indices) > 0:
        try:
            centers_rhs = torch.stack([viewpoint_stack[i].camera_center for i in rhs_indices], dim=0).detach()
            centers_lhs = torch.stack([viewpoint_stack[i].camera_center for i in lhs_indices], dim=0).detach()

            rhs_min, _ = centers_rhs.min(dim=0)
            rhs_max, _ = centers_rhs.max(dim=0)
            lhs_min, _ = centers_lhs.min(dim=0)
            lhs_max, _ = centers_lhs.max(dim=0)

            rhs_vol = float(((rhs_max - rhs_min).clamp(min=0.0).prod()).item())
            lhs_vol = float(((lhs_max - lhs_min).clamp(min=0.0).prod()).item())

            if rhs_vol > 0.0:
                lhs_coverage_pct = float(lhs_vol / rhs_vol)
        except Exception:
            lhs_coverage_pct = None
    elif not enable_ssgn:
        lhs_coverage_pct = 1.0

    if enable_ssgn:
        out_dict = linear_solve_ssgn(
            gaussians,
            opt,
            pipe,
            background,
            gsgn,
            imgs_rhs,
            imgs_lhs_for_solve,
            lhs_weights=lhs_weights,
            lhs_downsample_scale=lhs_downsample_scale,
            trust_region_radius=trust_region_radius,
            timing_dict=timing_dict,
        )
    else:
        out_dict = linear_solve_ssgn(
            gaussians,
            opt,
            pipe,
            background,
            gsgn,
            imgs_rhs,
            imgs_rhs,
            lhs_weights=None,
            lhs_downsample_scale=lhs_downsample_scale,
            trust_region_radius=trust_region_radius,
            timing_dict=timing_dict,
        )

    x = out_dict["x"]
    data = out_dict["data"]

    with measure_time("extract_update", timing_dict):
        x_dict = GaussianRasterizer.extract_gaussian_parameters(x, data)
        prev_params = {
            "xyz": gaussians._xyz.clone(),
            "scaling": gaussians._scaling.clone(),
            "rotation": gaussians._rotation.clone(),
            "opacity": gaussians._opacity.clone(),
            "features_dc": gaussians._features_dc.clone(),
            "features_rest": gaussians.get_active_features_rest.clone(),
        }

    with measure_time("line_search", timing_dict):
        line_search_use_maximum_gamma = getattr(gsgn, "line_search_use_maximum_gamma", False)
        line_search_maximum_gamma = getattr(gsgn, "line_search_maximum_gamma", 0.0)
        if line_search_use_maximum_gamma and line_search_maximum_gamma > 0:
            line_search_gamma_reset_interval = getattr(gsgn, "line_search_gamma_reset_interval", 100)
            if (iteration % line_search_gamma_reset_interval) == 0:
                gamma = min(gsgn.line_search_initial_gamma, line_search_maximum_gamma * 5)
            else:
                gamma = line_search_maximum_gamma
        else:
            gamma = gsgn.line_search_initial_gamma

        alpha = gsgn.line_search_alpha

        def update_params(g):
            gaussians._xyz.copy_((prev_params["xyz"] - g * gsgn.scale_fac_xyz * x_dict["xyz"]).float())
            gaussians._scaling.copy_((prev_params["scaling"] - g * gsgn.scale_fac_scale * x_dict["scaling"]).float())
            gaussians._rotation.copy_(
                (prev_params["rotation"] - g * gsgn.scale_fac_rotation * x_dict["rotation"]).float()
            )
            gaussians._opacity.copy_((prev_params["opacity"] - g * gsgn.scale_fac_opacity * x_dict["opacity"]).float())
            gaussians._features_dc.copy_(
                (prev_params["features_dc"] - g * gsgn.scale_fac_features_dc * x_dict["features_dc"]).float()
            )
            gaussians.get_active_features_rest.copy_(
                (prev_params["features_rest"] - g * gsgn.scale_fac_features_rest * x_dict["features_rest"]).float()
            )

        line_search_images = list(imgs_lhs_for_solve)
        random.shuffle(line_search_images)
        num_line_search_images = max(1, int(len(line_search_images) * gsgn.perc_images_in_line_search))
        line_search_images = line_search_images[:num_line_search_images]

        prev_error = 1e15

        while True:
            update_params(gamma)
            loss = 0.5 * render_and_add_to_residual_norm(
                gaussians=gaussians,
                viewpoint_stack=line_search_images,
                opt=opt,
                pipe=pipe,
                background=background,
            )

            if loss > prev_error:
                gamma /= alpha
                break

            if gamma < 1e-10:
                break

            gamma = alpha * gamma
            prev_error = loss

        del line_search_images

        if hasattr(gsgn, "line_search_maximum_gamma"):
            gsgn.line_search_maximum_gamma = max(gamma, gsgn.line_search_maximum_gamma)

    with measure_time("lm_update", timing_dict):
        delta_x = -gamma * x
        forward_output_for_check = out_dict["forward_output"]

        with measure_time("F_prev", timing_dict):
            update_params(0)
            forward_output_for_check.residuals = forward_output_for_check.residuals.to(
                delta_x.device,
                non_blocking=False,
            )
            if gsgn.compute_ssim_weights:
                forward_output_for_check.residuals_ssim = forward_output_for_check.residuals_ssim.to(
                    delta_x.device,
                    non_blocking=False,
                )

            F_prev = get_residual_norm(forward_output_for_check.residuals)
            if gsgn.compute_ssim_weights:
                F_prev += get_residual_norm(forward_output_for_check.residuals_ssim)

        with measure_time("F_new", timing_dict):
            update_params(gamma)
            F_new = render_and_add_to_residual_norm(
                gaussians=gaussians,
                viewpoint_stack=imgs_lhs_for_solve,
                opt=opt,
                pipe=pipe,
                background=background,
                compute_huber_weights=gsgn.compute_huber_weights,
                huber_c=gsgn.huber_c,
                compute_ssim_residuals=gsgn.compute_ssim_weights,
                ssim_residual_scale_factor=gsgn.ssim_residual_scale_factor,
            )
            if not isinstance(F_new, torch.Tensor):
                device = F_prev.device if isinstance(F_prev, torch.Tensor) else delta_x.device
                F_new = torch.tensor(F_new, device=device)

        cost_change = F_prev - F_new
        if not isinstance(cost_change, torch.Tensor):
            device = F_prev.device if isinstance(F_prev, torch.Tensor) else (
                F_new.device if isinstance(F_new, torch.Tensor) else delta_x.device
            )
            cost_change = torch.tensor(cost_change, device=device)

        terminate = False
        success = cost_change > 0

        if success:
            absolute_function_tolerance = F_prev * gsgn.function_tolerance
            if cost_change <= absolute_function_tolerance:
                terminate = True

            step_quality = 0.5
            min_factor = 1.0 / 3.0
            tmp_factor = 1.0 - (2 * step_quality - 1) ** 3
            trust_region_radius = trust_region_radius / max(min_factor, tmp_factor)
            trust_region_radius = min(trust_region_radius, gsgn.max_trust_region_radius)
            trust_region_radius = max(trust_region_radius, gsgn.min_trust_region_radius)
            radius_decrease_factor = 2.0
        else:
            print(f"no success, cost_change {cost_change}")
            trust_region_radius = trust_region_radius / radius_decrease_factor
            if cost_change < 0:
                print("reset to before iteration")
                update_params(0)
                radius_decrease_factor = 2 * radius_decrease_factor

            if trust_region_radius <= gsgn.min_trust_region_radius:
                trust_region_radius = gsgn.min_trust_region_radius
                radius_decrease_factor = 2
                terminate = True

    del prev_params
    del x
    del x_dict
    del delta_x

    out_dict["log_info"]["terminate"] = terminate
    out_dict["log_info"]["success"] = success
    out_dict["log_info"]["cost_change"] = cost_change
    out_dict["log_info"]["model_cost_change"] = torch.tensor(0.0, device=cost_change.device)
    out_dict["log_info"]["gamma"] = (
        gamma if isinstance(gamma, torch.Tensor) else torch.tensor(float(gamma), device=cost_change.device)
    )
    out_dict["log_info"]["lhs_coverage_pct"] = lhs_coverage_pct

    out_dict["loss"] = F_new
    out_dict["trust_region_radius"] = trust_region_radius
    out_dict["radius_decrease_factor"] = radius_decrease_factor
    out_dict["next_indices"] = viewpoint_stack_indices

    return out_dict

