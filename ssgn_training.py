import csv
import json
import os
import sys
import time
from random import randint
import random
import torch
from tqdm.auto import tqdm

from utils.loss_utils import l1_loss, ssim, l2_loss
from utils.image_utils import psnr
from train import (
    ModelParams,
    OptimizationParams,
    PipelineParams,
    GSGNParams,
    GaussianModel,
    Scene,
    render,
    network_gui,
    safe_state,
    prepare_output_and_logger,
    render_sets,
    evaluate,
    GaussianRasterizer,
    measure_time,
)
from ssgn_core import lm_step_ssgn


# ==============================================================================
# 6. Evaluation & logging utilities (with JSON/CSV saving)
# ==============================================================================
@torch.no_grad()
def compute_psnr_periodic(scene, renderFunc, renderArgs, iteration: int):
    torch.cuda.empty_cache()
    test_cameras = scene.getTestCameras()

    if not test_cameras or len(test_cameras) == 0:
        return None

    psnr_test = 0.0
    for viewpoint in test_cameras:
        image = torch.clamp(renderFunc(viewpoint, scene.gaussians, *renderArgs)["render"], 0.0, 1.0)
        gt_image = torch.clamp(viewpoint.original_image.to("cuda"), 0.0, 1.0)
        psnr_test += psnr(image.unsqueeze(0), gt_image.unsqueeze(0)).mean().double().item()

    psnr_test /= len(test_cameras)
    return psnr_test


@torch.no_grad()
def log_render_stats(tb_writer, iteration, testing_iterations, scene, renderFunc, renderArgs):
    torch.cuda.empty_cache()
    validation_configs = (
        {"name": "test", "cameras": scene.getTestCameras()},
        {"name": "train", "cameras": scene.getTrainCameras()},
    )

    results_dict = {}

    for config in validation_configs:
        if config["cameras"] and len(config["cameras"]) > 0:
            l1_test = 0.0
            l2_test = 0.0
            psnr_test = 0.0
            ssim_test = 0.0

            for idx, viewpoint in enumerate(config["cameras"]):
                image = torch.clamp(renderFunc(viewpoint, scene.gaussians, *renderArgs)["render"], 0.0, 1.0)
                gt_image = torch.clamp(viewpoint.original_image.to("cuda"), 0.0, 1.0)
                if tb_writer and (idx < 5):
                    tb_writer.add_images(
                        config["name"] + "_view_{}/render".format(viewpoint.image_name),
                        image[None],
                        global_step=iteration,
                    )
                    if iteration == testing_iterations[0]:
                        tb_writer.add_images(
                            config["name"] + "_view_{}/ground_truth".format(viewpoint.image_name),
                            gt_image[None],
                            global_step=iteration,
                        )
                l1_test += l1_loss(image, gt_image).mean().double().item()
                l2_test += l2_loss(image, gt_image).mean().double().item()
                psnr_test += psnr(image.unsqueeze(0), gt_image.unsqueeze(0)).mean().double().item()
                ssim_test += ssim(image.unsqueeze(0), gt_image.unsqueeze(0)).mean().double().item()

            psnr_test /= len(config["cameras"])
            l1_test /= len(config["cameras"])
            l2_test /= len(config["cameras"])
            ssim_test /= len(config["cameras"])

            print(
                "\n[ITER {}] Evaluating {}: L1 {} L2 {} PSNR {} SSIM {}".format(
                    iteration,
                    config["name"],
                    l1_test,
                    l2_test,
                    psnr_test,
                    ssim_test,
                )
            )

            if tb_writer:
                tb_writer.add_scalar(config["name"] + "/loss_viewpoint - l1_loss", l1_test, iteration)
                tb_writer.add_scalar(config["name"] + "/loss_viewpoint - l2_loss", l2_test, iteration)
                tb_writer.add_scalar(config["name"] + "/loss_viewpoint - psnr", psnr_test, iteration)
                tb_writer.add_scalar(config["name"] + "/loss_viewpoint - ssim", ssim_test, iteration)

            results_dict[config["name"]] = {
                "l1": l1_test,
                "l2": l2_test,
                "psnr": psnr_test,
                "ssim": ssim_test,
            }

    if results_dict:
        results_path = os.path.join(scene.model_path, f"results_test_{iteration}.json")
        try:
            with open(results_path, "w") as f:
                json.dump(results_dict, f, indent=2)
            print(f"[ITER {iteration}] Saved test results to {results_path}")
        except Exception as e:
            print(f"[WARNING] Failed to save test results: {e}")

    return results_dict


def training_report(
    tb_writer,
    iteration,
    out_dict,
    timing_dict,
    testing_iterations,
    scene,
    renderFunc,
    renderArgs,
    num_params,
    num_gaussians,
    image_width,
    image_height,
    psnr_history=None,
):
    if tb_writer:
        tb_writer.add_scalar("train/loss", out_dict["loss"].item(), iteration)
        tb_writer.add_scalars("time", timing_dict, iteration)
        timing_dict.clear()

        if "log_info" in out_dict:
            tb_writer.add_scalar("sparse_j/total_size_in_gb", out_dict["log_info"]["total_size_in_gb"], iteration)
            tb_writer.add_scalar("sparse_j/num_params", num_params, iteration)
            tb_writer.add_scalar("sparse_j/num_gaussians", num_gaussians, iteration)

            tb_writer.add_scalar("lm/radius_decrease_factor", out_dict["radius_decrease_factor"], iteration)
            tb_writer.add_scalar("lm/success", out_dict["log_info"]["success"], iteration)

    if iteration in testing_iterations:
        torch.cuda.empty_cache()

        validation_psnr = 0.0
        validation_l1 = 0.0

        check_cameras = scene.getTrainCameras()

        eval_subset = check_cameras[:5]

        if len(eval_subset) > 0:
            print(f"[ITER {iteration}] Calculating PSNR on {len(eval_subset)} training images...")

            for idx, viewpoint in enumerate(eval_subset):
                render_pkg = renderFunc(viewpoint, scene.gaussians, *renderArgs)
                image = torch.clamp(render_pkg["render"], 0.0, 1.0)
                gt_image = torch.clamp(viewpoint.original_image.to("cuda"), 0.0, 1.0)

                validation_psnr += psnr(image, gt_image).mean().double()
                validation_l1 += l1_loss(image, gt_image).mean().double()

                if tb_writer and idx == 0:
                    tb_writer.add_images("train_view/render", image[None], global_step=iteration)
                    if iteration == testing_iterations[0]:
                        tb_writer.add_images("train_view/ground_truth", gt_image[None], global_step=iteration)

            validation_psnr /= len(eval_subset)
            validation_l1 /= len(eval_subset)

            psnr_value = float(validation_psnr)
            l1_value = float(validation_l1)
        else:
            print("[WARNING] No cameras found for evaluation.")
            psnr_value = None
            l1_value = None

        if psnr_history is not None and psnr_value is not None:
            existing_record = next((r for r in psnr_history if r.get("iteration") == iteration), None)

            if existing_record:
                existing_record["psnr"] = psnr_value
                existing_record["l1"] = l1_value
            else:
                psnr_history.append(
                    {
                        "iteration": iteration,
                        "psnr": psnr_value,
                        "l1": l1_value,
                        "source": "train_set",
                    }
                )

            print(f"[ITER {iteration}] HISTORY UPDATED: PSNR={psnr_value:.4f}, L1={l1_value:.4f}")

            psnr_history_path = os.path.join(scene.model_path, "psnr_history.json")
            try:
                with open(psnr_history_path, "w") as f:
                    json.dump(psnr_history, f, indent=2)
            except Exception as e:
                print(f"Failed to save JSON: {e}")

            psnr_history_csv_path = os.path.join(scene.model_path, "psnr_history.csv")
            try:
                file_exists = os.path.isfile(psnr_history_csv_path)

                with open(psnr_history_csv_path, "a", newline="", encoding="utf-8") as f:
                    writer = csv.writer(f)

                    if not file_exists:
                        writer.writerow(["iteration", "psnr", "l1"])

                    def safe_format(val):
                        if val is None:
                            return ""
                        try:
                            return f"{float(val):.6f}"
                        except (ValueError, TypeError):
                            return str(val)

                    if psnr_history:
                        latest_record = psnr_history[-1]

                        if latest_record.get("iteration") == iteration:
                            writer.writerow(
                                [
                                    latest_record.get("iteration"),
                                    safe_format(latest_record.get("psnr")),
                                    safe_format(latest_record.get("l1")),
                                ]
                            )

            except Exception as e:
                print(f"[WARNING] Failed to save CSV: {e}")

        torch.cuda.empty_cache()


# ==============================================================================
# 7. SSGN training loop
# ==============================================================================
def training_ssgn(
    dataset,
    opt,
    gsgn,
    pipe,
    testing_iterations,
    saving_iterations,
    checkpoint_iterations,
    checkpoint,
    debug_from,
    quiet,
):
    print("\n[SSGN] Starting Training with SSGN + FPS...")

    training_start_time = time.time()

    first_iter = 0
    tb_writer = prepare_output_and_logger(dataset, opt, gsgn, pipe, quiet)
    gaussians = GaussianModel(dataset.sh_degree)
    
    # Debug: Print source_path before Scene initialization
    print(f"[DEBUG] dataset.source_path = {dataset.source_path}")
    print(f"[DEBUG] Checking if source_path exists: {os.path.exists(dataset.source_path)}")
    if os.path.exists(dataset.source_path):
        sparse_path = os.path.join(dataset.source_path, "sparse")
        transforms_path = os.path.join(dataset.source_path, "transforms_train.json")
        print(f"[DEBUG] sparse/ exists: {os.path.exists(sparse_path)}")
        print(f"[DEBUG] transforms_train.json exists: {os.path.exists(transforms_path)}")
    
    scene = Scene(dataset, gaussians)

    import train as _train_ori_mod

    if hasattr(_train_ori_mod, "args") and getattr(_train_ori_mod.args, "resume_from_ply", None):
        ply_path = _train_ori_mod.args.resume_from_ply
        print("[RESUME_FROM_PLY] loading:", ply_path)
        assert os.path.isfile(ply_path), f"PLY not found: {ply_path}"
        gaussians.load_ply(ply_path)
        try:
            gsgn.num_sgd_iterations_before_gn = 0
        except Exception:
            pass
        print("[RESUME_FROM_PLY] done. num_gaussians=", len(gaussians.get_xyz))

    gaussians.training_setup(opt)

    trust_region_radius = gsgn.trust_region_radius
    radius_decrease_factor = gsgn.radius_decrease_factor

    bg_color = [1, 1, 1] if dataset.white_background else [0, 0, 0]
    background = torch.tensor(bg_color, dtype=torch.float32, device="cuda")

    viewpoint_stack = scene.getTrainCameras().copy()
    viewpoint_stack_indices = list(range(len(viewpoint_stack)))
    random.shuffle(viewpoint_stack_indices)

    viewpoint_stack_sgd = None

    def sgd_step(viewpoint_stack_local, iteration_local: int = -1):
        if iteration_local > -1:
            gaussians.update_learning_rate(iteration_local)

            if opt.increase_SH_deg_every > 0 and (iteration_local % opt.increase_SH_deg_every) == 0:
                gaussians.oneupSHdegree()

        viewpoint_cam = viewpoint_stack_local.pop(randint(0, len(viewpoint_stack_local) - 1))

        bg = torch.rand((3), device="cuda") if opt.random_background else background
        render_pkg = render(viewpoint_cam, gaussians, pipe, bg)
        image, viewspace_point_tensor, visibility_filter, radii = (
            render_pkg["render"],
            render_pkg["viewspace_points"],
            render_pkg["visibility_filter"],
            render_pkg["radii"],
        )

        gt_image = viewpoint_cam.original_image.cuda()
        Ll1 = l1_loss(image, gt_image)
        loss = opt.lambda_l1 * Ll1 + opt.lambda_dssim * (1.0 - ssim(image, gt_image))
        if opt.lambda_l2 > 0:
            Ll2 = l2_loss(image, gt_image)
            loss = loss + opt.lambda_l2 * Ll2
        loss.backward()

        with torch.no_grad():
            if iteration_local > -1 and iteration_local < opt.densify_until_iter:
                gaussians.max_radii2D[visibility_filter] = torch.max(
                    gaussians.max_radii2D[visibility_filter],
                    radii[visibility_filter],
                )
                gaussians.add_densification_stats(viewspace_point_tensor, visibility_filter)

                if iteration_local > opt.densify_from_iter and iteration_local % opt.densification_interval == 0:
                    size_threshold = 20 if iteration_local > opt.opacity_reset_interval else None
                    (
                        n_points_before_densification,
                        n_points_after_densification,
                        n_points_after_prune,
                    ) = gaussians.densify_and_prune(
                        opt.densify_grad_threshold,
                        0.005,
                        scene.cameras_extent,
                        size_threshold,
                    )
                    if tb_writer is not None:
                        tb_writer.add_scalars(
                            "densification",
                            {
                                "before_densification": n_points_before_densification,
                                "after_densification": n_points_after_densification,
                                "after_prune": n_points_after_prune,
                            },
                            iteration_local,
                        )

                if iteration_local % opt.opacity_reset_interval == 0 or (
                    dataset.white_background and iteration_local == opt.densify_from_iter
                ):
                    gaussians.reset_opacity()

        gaussians.optimizer.step()
        gaussians.optimizer.zero_grad(set_to_none=True)

    if opt.iterations == 0:
        print("\n[SSGN] iterations=0: Only performing SGD initialization, skipping LM optimization...")
        if gsgn.num_sgd_iterations_before_gn > 0:
            print(f"[SSGN] Running {gsgn.num_sgd_iterations_before_gn} SGD iterations...")
            for i in tqdm(
                range(gsgn.num_sgd_iterations_before_gn),
                desc="sgd_before_gn",
                leave=True,
                miniters=100,
                mininterval=5.0,
            ):
                if not viewpoint_stack_sgd:
                    viewpoint_stack_sgd = scene.getTrainCameras().copy()
                sgd_step(viewpoint_stack_sgd, 1 + i)
            print("[SSGN] SGD initialization completed")

            save_iteration = gsgn.num_sgd_iterations_before_gn
            print(f"\n[ITER {save_iteration}] Saving Gaussians after SGD initialization...")
            scene.save(save_iteration)

            try:
                mem = torch.cuda.max_memory_allocated() / 1024 ** 3
                stats = {
                    "mem": mem,
                    "elapsed_time": time.time() - training_start_time,
                    "num_GS": len(gaussians.get_xyz),
                    "num_gaussians": len(gaussians.get_xyz),
                    "lhs_coverage_pct": None,
                    "frame_selection_method": "SGD_Init",
                }
                stats_path = os.path.join(scene.model_path, f"train_stats_{save_iteration}.json")
                with open(stats_path, "w") as f:
                    json.dump(stats, f, indent=2)
                print(f"[ITER {save_iteration}] Saved train stats to {stats_path}")

                log_render_stats(
                    tb_writer,
                    save_iteration,
                    [save_iteration],
                    scene,
                    render,
                    (pipe, background),
                )
            except Exception as e:
                print(f"[ERROR] Failed to save stats for iter 0: {e}")

            print(f"[ITER {save_iteration}] Model saved successfully")
        else:
            print("[WARNING] iterations=0 but num_sgd_iterations_before_gn=0, nothing to do.")
        print("Training complete (SGD initialization only).")
        return scene

    progress_bar = tqdm(range(first_iter, opt.iterations), desc="Training progress")
    first_iter += 1

    timing_dict = {}

    psnr_history = []
    psnr_history_path = os.path.join(scene.model_path, "psnr_history.json")
    psnr_history_csv_path = os.path.join(scene.model_path, "psnr_history.csv")

    for iteration in range(first_iter, opt.iterations + 1):
        if not quiet:
            if network_gui.conn is None:
                network_gui.try_connect()
            while network_gui.conn is not None:
                try:
                    net_image_bytes = None
                    (
                        custom_cam,
                        do_training,
                        pipe.convert_SHs_python,
                        pipe.compute_cov3D_python,
                        keep_alive,
                        scaling_modifer,
                    ) = network_gui.receive()
                    if custom_cam is not None:
                        net_image = render(custom_cam, gaussians, pipe, background, scaling_modifer)["render"]
                        net_image_bytes = memoryview(
                            (torch.clamp(net_image, min=0, max=1.0) * 255)
                            .byte()
                            .permute(1, 2, 0)
                            .contiguous()
                            .cpu()
                            .numpy()
                        )
                    network_gui.send(net_image_bytes, dataset.source_path)
                    if do_training and ((iteration < int(opt.iterations)) or not keep_alive):
                        break
                except Exception:
                    network_gui.conn = None

        if gsgn.num_sgd_iterations_before_gn > 0 and iteration == 1:
            print(f"\n[SSGN] Running {gsgn.num_sgd_iterations_before_gn} SGD iterations before LM optimization...")
            for i in tqdm(
                range(gsgn.num_sgd_iterations_before_gn),
                desc="sgd_before_gn",
                leave=True,
                miniters=100,
                mininterval=5.0,
            ):
                with measure_time("sgd_before_gn", timing_dict, additive=True):
                    if not viewpoint_stack_sgd:
                        viewpoint_stack_sgd = scene.getTrainCameras().copy()
                    sgd_step(viewpoint_stack_sgd, 1 + i)
            print("[SSGN] SGD initialization completed")
            sys.stdout.flush()

            if hasattr(gsgn, "num_sgd_iterations_between_gn") and hasattr(gsgn, "num_sgd_iterations_after_gn"):
                if gsgn.num_sgd_iterations_between_gn <= 0 and gsgn.num_sgd_iterations_after_gn <= 0 and len(
                    checkpoint_iterations
                ) == 0:
                    print("[SSGN] Removing SGD data to free memory...")
                    gaussians.remove_sgd_data()
                    print("[SSGN] SGD data removed")
                    sys.stdout.flush()

        lhs_downsample_scale = getattr(gsgn, "lhs_downsample_scale", 1)

        out_dict = lm_step_ssgn(
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
            None,
            output_path=scene.model_path,
            lhs_downsample_scale=lhs_downsample_scale,
        )

        trust_region_radius = out_dict["trust_region_radius"]
        radius_decrease_factor = out_dict["radius_decrease_factor"]
        viewpoint_stack_indices = out_dict["next_indices"]

        progress_bar.set_postfix({"Loss": f"{out_dict['loss']:.7f}"})
        progress_bar.update(1)

        if iteration % 1 == 0:
            try:
                metrics = compute_psnr_periodic(scene, render, (pipe, background), iteration)

                if isinstance(metrics, tuple):
                    psnr_value, l1_value = metrics
                else:
                    psnr_value = metrics
                    l1_value = None

                if psnr_value is not None:
                    record = {"iteration": iteration, "psnr": float(psnr_value)}
                    if l1_value is not None:
                        record["l1"] = float(l1_value)

                    psnr_history.append(record)

                    l1_str = f", L1: {l1_value:.4f}" if l1_value is not None else ""
                    print(f"[ITER {iteration}] PSNR: {psnr_value:.4f}{l1_str}")

                    try:
                        with open(psnr_history_path, "w") as f:
                            json.dump(psnr_history, f, indent=2)
                    except Exception as e:
                        print(f"[WARNING] Failed to save PSNR history to JSON: {e}")

                    try:
                        file_exists = os.path.exists(psnr_history_csv_path)

                        with open(psnr_history_csv_path, "a", newline="", encoding="utf-8") as f:
                            writer = csv.writer(f)
                            if not file_exists:
                                writer.writerow(["iteration", "psnr", "l1"])

                            def safe_fmt(val):
                                if val is None:
                                    return ""
                                try:
                                    return f"{float(val):.6f}"
                                except (ValueError, TypeError):
                                    return str(val)

                            writer.writerow([iteration, safe_fmt(psnr_value), safe_fmt(l1_value)])
                    except Exception as e:
                        print(f"[WARNING] Failed to save PSNR history to CSV: {e}")

            except Exception as e:
                print(f"[WARNING] Failed to compute/save metrics at iteration {iteration}: {e}")

        if not quiet:
            training_report(
                tb_writer,
                iteration,
                out_dict,
                timing_dict,
                testing_iterations,
                scene,
                render,
                (pipe, background),
                gaussians.total_params,
                gaussians.num_gaussians,
                viewpoint_stack[0].image_width,
                viewpoint_stack[0].image_height,
                psnr_history=psnr_history,
            )

        if iteration in saving_iterations:
            print(f"\n[ITER {iteration}] Saving Gaussians")
            try:
                mem = torch.cuda.max_memory_allocated() / 1024 ** 3

                lhs_coverage_pct = None
                if "log_info" in out_dict and "lhs_coverage_pct" in out_dict["log_info"]:
                    lhs_coverage_pct = out_dict["log_info"]["lhs_coverage_pct"]

                enable_fps_lhs = getattr(gsgn, "enable_fps_lhs", False)
                if enable_fps_lhs:
                    frame_selection_method = "FPS"
                else:
                    frame_selection_method = getattr(gsgn, "image_subsample_frame_selection_mode", "strided")

                stats = {
                    "mem": mem,
                    "elapsed_time": time.time() - training_start_time,
                    "num_GS": len(gaussians.get_xyz),
                    "num_gaussians": len(gaussians.get_xyz),
                    "lhs_coverage_pct": lhs_coverage_pct,
                    "frame_selection_method": frame_selection_method,
                }
                stats_path = os.path.join(scene.model_path, f"train_stats_{iteration}.json")
                with open(stats_path, "w") as f:
                    json.dump(stats, f, indent=2)
                print(f"[ITER {iteration}] Saved train stats to {stats_path}")
                if lhs_coverage_pct is not None:
                    print(f"[ITER {iteration}] LHS coverage: {lhs_coverage_pct:.4f}")

                scene.save(iteration)
                ply_path = os.path.join(
                    scene.model_path,
                    "point_cloud",
                    f"iteration_{iteration}",
                    "point_cloud.ply",
                )
                if os.path.exists(ply_path):
                    print(f"[ITER {iteration}] Successfully saved model to {ply_path}")
                else:
                    print(f"[WARNING] Model file not found at {ply_path} after save operation")
            except Exception as e:
                print(f"[ERROR] Failed to save model at iteration {iteration}: {e}")
                import traceback

                traceback.print_exc()

    if psnr_history:
        try:
            with open(psnr_history_path, "w") as f:
                json.dump(psnr_history, f, indent=2)
            print(
                f"\n[Training Complete] Saved PSNR history to {psnr_history_path} ({len(psnr_history)} records)"
            )

            with open(psnr_history_csv_path, "w", newline="", encoding="utf-8") as f:
                writer = csv.writer(f)
                writer.writerow(["iteration", "psnr", "l1"])
                for record in psnr_history:
                    iteration_val = record.get("iteration", "")
                    psnr_val = f"{record.get('psnr', 0):.6f}"
                    l1_val = f"{record.get('l1', 0):.6f}" if record.get("l1") is not None else ""
                    writer.writerow([iteration_val, psnr_val, l1_val])
            print(
                f"[Training Complete] Saved PSNR history to {psnr_history_csv_path} ({len(psnr_history)} records)"
            )
        except Exception as e:
            print(f"[WARNING] Failed to save final PSNR history: {e}")

    print("Training complete.")
    return scene

