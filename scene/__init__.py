#
# Copyright (C) 2023, Inria
# GRAPHDECO research group, https://team.inria.fr/graphdeco
# All rights reserved.
#
# This software is free for non-commercial, research and evaluation use 
# under the terms of the LICENSE.md file.
#
# For inquiries contact  george.drettakis@inria.fr
#

import os
import random
import json
from utils.system_utils import searchForMaxIteration
from scene.dataset_readers import sceneLoadTypeCallbacks
from scene.gaussian_model import GaussianModel
from arguments import ModelParams
from utils.camera_utils import cameraList_from_camInfos, camera_to_JSON

class Scene:

    gaussians : GaussianModel

    def __init__(self, args : ModelParams, gaussians : GaussianModel, load_iteration=None, shuffle=True, resolution_scales=[1.0], loaded_cams=None):
        """b
        :param path: Path to colmap scene main folder.
        """
        self.model_path = args.model_path
        self.loaded_iter = None
        self.gaussians = gaussians

        if load_iteration:
            if load_iteration == -1:
                self.loaded_iter = searchForMaxIteration(os.path.join(self.model_path, "point_cloud"))
            else:
                self.loaded_iter = load_iteration
            print("Loading trained model at iteration {}".format(self.loaded_iter))

        if loaded_cams is None:
            self.train_cameras = {}
            self.test_cameras = {}
            self.spherical_cameras = {}

            if os.path.exists(os.path.join(args.source_path, "sparse")):
                scene_info = sceneLoadTypeCallbacks["Colmap"](args.source_path, args.images, args.eval, points_pcl_suffix=args.points_pcl_suffix)
            elif os.path.exists(os.path.join(args.source_path, "transforms_train.json")):
                print("Found transforms_train.json file, assuming Blender data set!")
                scene_info = sceneLoadTypeCallbacks["Blender"](args.source_path, args.white_background, args.eval, points_pcl_suffix=args.points_pcl_suffix if hasattr(args, "points_pcl_suffix") else "")
            else:
                assert False, f"Could not recognize scene type at {args.source_path}!"

            if not self.loaded_iter:
                with open(scene_info.ply_path, 'rb') as src_file, open(os.path.join(self.model_path, "input.ply") , 'wb') as dest_file:
                    dest_file.write(src_file.read())
                json_cams = []
                camlist = []
                if scene_info.test_cameras:
                    camlist.extend(scene_info.test_cameras)
                if scene_info.train_cameras:
                    camlist.extend(scene_info.train_cameras)
                for id, cam in enumerate(camlist):
                    json_cams.append(camera_to_JSON(id, cam))
                with open(os.path.join(self.model_path, "cameras.json"), 'w') as file:
                    json.dump(json_cams, file)

            if shuffle:
                random.shuffle(scene_info.train_cameras)  # Multi-res consistent random shuffling
                # random.shuffle(scene_info.test_cameras)  # Multi-res consistent random shuffling

            self.cameras_extent = scene_info.nerf_normalization["radius"]

            for resolution_scale in resolution_scales:
                print("Loading Training Cameras")
                self.train_cameras[resolution_scale] = cameraList_from_camInfos(scene_info.train_cameras, resolution_scale, args)
                print("Loading Test Cameras (to cpu)")
                old_data_device = args.data_device
                args.data_device = "cpu"
                self.test_cameras[resolution_scale] = cameraList_from_camInfos(scene_info.test_cameras, resolution_scale, args)
                if scene_info.spherical_cameras is not None:
                    print("Loading Spherical Cameras (to cpu)")
                    self.spherical_cameras[resolution_scale] = cameraList_from_camInfos(scene_info.spherical_cameras, resolution_scale, args)
                else:
                    print("Do not have spherical cameras")
                args.data_device = old_data_device
        else:
            self.train_cameras = loaded_cams["train"]
            self.test_cameras = loaded_cams["test"]
            self.spherical_cameras = loaded_cams["spherical"]
            self.cameras_extent = loaded_cams["extent"]
            # 如果提供了 loaded_cams，但没有 loaded_iter，说明是从训练中传递过来的
            # 此时应该已经有保存的模型了，尝试查找最新的迭代
            if not self.loaded_iter:
                self.loaded_iter = searchForMaxIteration(os.path.join(self.model_path, "point_cloud"))

        if self.loaded_iter:
            self.gaussians.load_ply(os.path.join(self.model_path,
                                                           "point_cloud",
                                                           "iteration_" + str(self.loaded_iter),
                                                           "point_cloud.ply"))
        else:
            # 只有在 loaded_cams 为 None 时，scene_info 才会被定义
            if loaded_cams is None:
                self.gaussians.create_from_pcd(scene_info.point_cloud, self.cameras_extent, dist2_path=scene_info.dist2_path)
            else:
                # 如果 loaded_cams 不为 None 但没有 loaded_iter，说明模型还没有被保存
                # 这种情况下，应该从输入点云加载（如果存在）
                input_ply_path = os.path.join(self.model_path, "input.ply")
                if os.path.exists(input_ply_path):
                    # 从保存的 input.ply 加载
                    self.gaussians.load_ply(input_ply_path)
                else:
                    # 如果 input.ply 也不存在，尝试从原始路径加载
                    if os.path.exists(os.path.join(args.source_path, "sparse")):
                        scene_info = sceneLoadTypeCallbacks["Colmap"](args.source_path, args.images, args.eval, points_pcl_suffix=args.points_pcl_suffix)
                    elif os.path.exists(os.path.join(args.source_path, "transforms_train.json")):
                        scene_info = sceneLoadTypeCallbacks["Blender"](args.source_path, args.white_background, args.eval, points_pcl_suffix=args.points_pcl_suffix if hasattr(args, "points_pcl_suffix") else "")
                    else:
                        raise RuntimeError(f"Could not find point cloud to load. Please ensure model is saved or input.ply exists.")
                    self.gaussians.create_from_pcd(scene_info.point_cloud, self.cameras_extent, dist2_path=scene_info.dist2_path)

    def save(self, iteration):
        point_cloud_path = os.path.join(self.model_path, "point_cloud/iteration_{}".format(iteration))
        self.gaussians.save_ply(os.path.join(point_cloud_path, "point_cloud.ply"))

    def getTrainCameras(self, scale=1.0):
        return self.train_cameras[scale]

    def getTestCameras(self, scale=1.0):
        return self.test_cameras[scale]

    def hasSphericalCameras(self):
        return len(self.spherical_cameras.keys()) > 0

    def getSphericalCameras(self, scale=1.0):
        return self.spherical_cameras[scale]
