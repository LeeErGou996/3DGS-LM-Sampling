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
import json
from argparse import ArgumentParser, Namespace
try:
    from typing import Literal
except ImportError:
    from typing_extensions import Literal
import sys
import os


class GroupParams:
    pass


class ParamGroup:
    def __init__(self, parser: ArgumentParser, name : str, fill_none = False):
        group = parser.add_argument_group(name)
        for key, value in vars(self).items():
            shorthand = False
            if key.startswith("_"):
                shorthand = True
                key = key[1:]
            t = type(value)
            value = value if not fill_none else None 
            if shorthand:
                if t == bool:
                    if value is True:
                        # 对于默认值为 True 的布尔参数，添加 --no_ 选项来设置为 False
                        group.add_argument("--" + key, ("-" + key[0:1]), default=value, action="store_true",
                                         help=f"Enable {key} (default: True)")
                        group.add_argument("--no_" + key, dest=key, action="store_false",
                                         help=f"Disable {key}")
                    else:
                        group.add_argument("--" + key, ("-" + key[0:1]), default=value, action="store_true")
                else:
                    group.add_argument("--" + key, ("-" + key[0:1]), default=value, type=t)
            else:
                if t == bool:
                    if value is True:
                        # 对于默认值为 True 的布尔参数，添加 --no_ 选项来设置为 False
                        group.add_argument("--" + key, default=value, action="store_true", 
                                         help=f"Enable {key} (default: True)")
                        group.add_argument("--no_" + key, dest=key, action="store_false",
                                         help=f"Disable {key}")
                    else:
                        # 对于默认值为 False 的布尔参数，使用 store_true
                        group.add_argument("--" + key, default=value, action="store_true")
                else:
                    group.add_argument("--" + key, default=value, type=t)

    def extract(self, args):
        group = GroupParams()
        for arg in vars(args).items():
            if arg[0] in vars(self) or ("_" + arg[0]) in vars(self):
                setattr(group, arg[0], arg[1])
        return group


class ModelParams(ParamGroup): 
    def __init__(self, parser, sentinel=False):
        self.sh_degree = 3
        self.root_out = "./output"
        self.exp_name = ""
        self._source_path = ""
        self._model_path = ""
        self._images = "images"
        self._resolution = -1
        self._white_background = False
        self.data_device = "cuda"
        self.eval = False
        self.points_pcl_suffix: str = ""
        super().__init__(parser, "Loading Parameters", sentinel)

    def extract(self, args):
        g = super().extract(args)
        g.source_path = os.path.abspath(g.source_path)
        return g


class PipelineParams(ParamGroup):
    def __init__(self, parser):
        self.convert_SHs_python = False
        self.compute_cov3D_python = False
        self.debug = False
        super().__init__(parser, "Pipeline Parameters")


class OptimizationParams(ParamGroup):
    def __init__(self, parser):
        # =========================================================
        # 这里保持 SGD 的默认值 (30,000步)，确保原版代码逻辑不受影响
        # =========================================================
        self.iterations = 30_000
        self.position_lr_init = 0.00016
        self.position_lr_final = 0.0000016
        self.position_lr_delay_mult = 0.01
        self.position_lr_max_steps = 30_000
        self.feature_lr = 0.0025
        self.opacity_lr = 0.05
        self.scaling_lr = 0.005
        self.rotation_lr = 0.001
        self.percent_dense = 0.01
        self.lambda_dssim = 0.2
        self.lambda_l1 = 0.8
        self.lambda_l2 = 0.0
        
        # === 密度控制相关参数 (SGD 默认值) ===
        self.densification_interval = 100
        self.opacity_reset_interval = 3000
        self.densify_from_iter = 500
        self.densify_until_iter = 15_000
        self.densify_grad_threshold = 0.0002 # 0.0002
        
        self.increase_SH_deg_every = 1000
        self.random_background = False
        super().__init__(parser, "Optimization Parameters")

    def adjust_for_lm(self, args, target_iterations):
        """
        [新增功能] 如果使用 LM 算法，按比例缩放时间表参数并更新 args 对象
        
        :param args: 解析后的命令行参数对象 (Namespace)
        :param target_iterations: LM 计划运行的总步数 (例如 1000)
        """
        # 1. 计算缩放因子 (SGD标准步数 / LM目标步数)
        # 例如: 30000 / 1000 = 30倍加速
        sgd_baseline = 30_000
        ratio = target_iterations / sgd_baseline

        print(f"\n[Auto-Config] 检测到 LM 模式 (Iter={target_iterations})")
        print(f"[Auto-Config] 正在按比例 (Ratio={ratio:.4f}) 缩放密度控制时间表...")

        # 2. 定义辅助函数：缩放并取整，同时保证最小值为 1
        def scale_param(val):
            return max(1, int(val * ratio))

        # 3. 更新 args 中的值 (这是真正生效的地方)
        
        # A. 更新分裂开始时间 (原500 -> 约16)
        old_start = args.densify_from_iter
        args.densify_from_iter = scale_param(500) 
        
        # B. 更新分裂结束时间 (原15000 -> 约500)
        old_end = args.densify_until_iter
        args.densify_until_iter = scale_param(15_000)
        
        # C. 更新分裂间隔 (原100 -> 约3)
        # 注意：LM中如果间隔太短(如<10)可能会太慢，所以这里加个下限保护
        raw_interval = 100 * ratio
        args.densification_interval = max(10, int(raw_interval)) 
        
        # D. 更新透明度重置间隔 (原3000 -> 约100)
        old_reset = args.opacity_reset_interval
        args.opacity_reset_interval = scale_param(3000)

        # E. 更新 SH 系数提升时间
        old_sh_increase = args.increase_SH_deg_every
        args.increase_SH_deg_every = scale_param(1000)
        
        # F. 更新位置学习率的最大步数
        old_lr_max_steps = args.position_lr_max_steps
        args.position_lr_max_steps = scale_param(30_000)

        # 打印修改结果对比
        print(f"  -> Densify Start:    {old_start}  -> {args.densify_from_iter}")
        print(f"  -> Densify End:      {old_end} -> {args.densify_until_iter}")
        print(f"  -> Densify Interval: {100}      -> {args.densification_interval}")
        print(f"  -> Opacity Reset:    {old_reset} -> {args.opacity_reset_interval}")
        print(f"  -> SH Increase:      {old_sh_increase}      -> {args.increase_SH_deg_every}")
        print(f"  -> LR Max Steps:     {old_lr_max_steps}  -> {args.position_lr_max_steps}")
        print("-" * 50)


class GSGNParams(ParamGroup):
    def __init__(self, parser):
        self.use_double_precision: bool = False
        self.trust_region_radius: float = 1.0
        self.min_trust_region_radius: float = 1e-4
        self.max_trust_region_radius: float = 1e4
        self.radius_decrease_factor: float = 2.0
        self.max_grad_norm: float = 1e1
        self.min_lm_diagonal: float = 1e0
        self.max_lm_diagonal: float = 1e6
        self.min_relative_decrease: float = 1e-5
        self.function_tolerance: float = 0.000001
        self.pcg_max_iter: int = 100
        self.pcg_rtol: float = 1e-6
        self.pcg_atol: float = 0.0
        self.pcg_gradient_descent_every: int = -1
        self.pcg_explicit_residual_every: int = -1
        self.pcg_verbose: bool = False
        self.perc_images_in_line_search: float = 1.0
        self.line_search_initial_gamma: float = 1.0 # 1.0
        self.line_search_alpha: float = 0.7
        self.line_search_use_maximum_gamma: bool = False
        self.line_search_maximum_gamma: float = -1.0
        self.line_search_gamma_reset_interval: int = 100
        self.num_sgd_iterations_before_gn: int = 0
        self.num_sgd_iterations_between_gn: int = 0
        self.sgd_between_gn_every: int = 0
        self.num_sgd_iterations_after_gn: int = 0
        self.image_subsample_size: int = -1
        self.image_subsample_n_iters: int = 1
        self.image_subsample_frame_selection_mode: Literal["random", "strided"] = "strided"
        self.average_pcg_mode: Literal["mean", "max", "diag_jtj"] = "diag_jtj"
        self.scale_fac_xyz: float = 1.0
        self.scale_fac_opacity: float = 1.0
        self.scale_fac_rotation: float = 1.0
        self.scale_fac_scale: float = 1.0
        self.scale_fac_features_dc: float = 1.0
        self.scale_fac_features_rest: float = 1.0
        self.compute_huber_weights: bool = True
        self.huber_c: float = 0.1
        self.compute_ssim_weights: bool = True
        self.ssim_residual_scale_factor: float = 0.25
        
        # ⬇️⬇️⬇️ 新增：SSGN 控制参数 ⬇️⬇️⬇️
        self.enable_ssgn: bool = True  # 开关：是否启用 SSGN
        self.ssgn_size_rhs: int = 25   # 大批次 (计算 b)
        self.ssgn_size_lhs: int = 25   # 小批次 (计算 J 缓存, 决定显存峰值)
        self.enable_fps_lhs: bool = False  # 开关：是否在 LHS 采样时使用 Farthest Point Sampling
        self.use_voronoi_weights: bool = False # 开关：是否在 LHS 采样时使用 Voronoi 权重
        self.no_resolution_scale: bool = False # 开关：是否禁用分辨率能量补偿
        self.sample_stride: int = -1 # [新增] 手动控制 LHS 采样的 stride，-1 表示自动计算 (RHS/LHS)
        self.lhs_ratios: float = 0.0  # LHS 采样比例，如果为 0 则只运行 SGD 初始化，跳过 LM 优化

        # ⬇️⬇️⬇️ 新增：初始化与优化方法控制参数 ⬇️⬇️⬇️
        # init_method: 决定 Gaussians 的初始化方式（纯 SGD 还是 EDGS corr_init）
        self.init_method: Literal["sgd", "corr_init"] = "sgd"
        # optimization_method: 初始化之后使用的全局优化器类型
        self.optimization_method: Literal["sgd", "lm"] = "lm"

        # ⬇️⬇️⬇️ 新增：corr_init 相关参数（参考 EDGS/configs/train.yaml） ⬇️⬇️⬇️
        # 是否启用基于 RoMa 的对应点初始化
        self.corr_init_use: bool = True
        # RoMa 模型类型：室内 / 室外；为兼容 EDGS，支持 "outdoors" 写法
        self.corr_init_roma_model: Literal["indoors", "outdoor", "outdoors"] = "outdoor"
        # 每个参考帧的匹配点数量
        self.corr_init_matches_per_ref: int = 15_000
        # 三角化时的缩放因子
        self.corr_init_scaling_factor: float = 0.001
        # 投影误差容差
        self.corr_init_proj_err_tolerance: float = 0.01
        # 参考帧数量 / 每个参考帧的最近邻数
        self.corr_init_num_refs: int = 180
        self.corr_init_nns_per_ref: int = 3
        # 是否保留原始 SfM 初始化点
        self.corr_init_add_sfm_init: bool = False
        # 是否在 corr_init 过程中打印详细日志
        self.corr_init_verbose: bool = False
        
        super().__init__(parser, "GN Parameters")


def get_combined_args(parser : ArgumentParser):
    cmdlne_string = sys.argv[1:]
    cfgfile_string = "Namespace()"
    args_cmdline = parser.parse_args(cmdlne_string)
    cfgfilepath = os.path.join(args_cmdline.model_path, "cfg_args")

    if not os.path.exists(cfgfilepath):
        cfgfilepath = cfgfilepath + ".json"
        merged_dict = {}
        with open(cfgfilepath) as cfg_file:
            cfg = json.load(cfg_file)
            for d in cfg.values():
                for k, v in d.items():
                    merged_dict[k] = v
    else:
        try:
            print("Looking for config file in", cfgfilepath)
            with open(cfgfilepath) as cfg_file:
                print("Config file found: {}".format(cfgfilepath))
                cfgfile_string = cfg_file.read()
        except TypeError:
            print("Config file not found at")
            pass
        args_cfgfile = eval(cfgfile_string)
        merged_dict = vars(args_cfgfile).copy()

    for k,v in vars(args_cmdline).items():
        if v != None:
            merged_dict[k] = v
    return Namespace(**merged_dict)
