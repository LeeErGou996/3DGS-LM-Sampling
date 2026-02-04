# Python functions

# Imports
import torch
import torch.nn as nn
from typing import Dict, List, Tuple
from contextlib import contextmanager
from dataclasses import dataclass, field
import math

# Try to import the compiled C++ extension
try:
    import rhs_cuda_extension._C_RHS as _C
except ImportError:
    print("Failed to import C++ extension. Please compile it first.")
    _C = None


# Define necessary data structures
@dataclass
class RenderedImageAndBackwardValues:
    num_images: int = 0
    H: int = 0
    W: int = 0
    bg: torch.Tensor = None
    scale_modifier: float = -1.0
    sh_degree: int = -1
    debug: bool = False
    viewmatrices: List = field(default_factory=list)
    projmatrices: List = field(default_factory=list)
    camposes: List = field(default_factory=list)
    tanfovxs: List = field(default_factory=list)
    tanfovys: List = field(default_factory=list)
    cxs: List = field(default_factory=list)
    cys: List = field(default_factory=list)
    geomBuffers: List = field(default_factory=list)
    binningBuffers: List = field(default_factory=list)
    imgBuffers: List = field(default_factory=list)
    num_rendered_list: List = field(default_factory=list)
    n_contrib_vol_rend: List = field(default_factory=list)
    is_gaussian_hit: List = field(default_factory=list)
    residuals: torch.Tensor = None
    weights: torch.Tensor = None
    residuals_ssim: torch.Tensor = None
    weights_ssim: torch.Tensor = None

# Mock context manager
@contextmanager
def measure_time(name: str, out_dict: Dict[str, float], additive: bool = False, maximum: bool = False):
    yield

def safe_call_fn(fn, args, debug):
    return fn(*args)

class GaussianRasterizer(nn.Module):
    def __init__(self):
        super().__init__()
    
    @staticmethod
    def build_gsgn_data_spec(
        forward_output: RenderedImageAndBackwardValues,
        params: torch.Tensor,
        means3D: torch.Tensor = None,
        shs: torch.Tensor = None,
        unactivated_opacities: torch.Tensor = None,
        unactivated_scales: torch.Tensor = None,
        unactivated_rotations: torch.Tensor = None,
        use_double_precision: bool = False,
    ):
        """
        Generate object to pass to C++
        """
        gsgn_data_spec = _C.GSGNDataSpec()
        gsgn_data_spec.background = forward_output.bg
        gsgn_data_spec.params = params.contiguous()
        gsgn_data_spec.means3D = means3D.contiguous() if means3D is not None else torch.empty(0, device="cuda")
        gsgn_data_spec.scale_modifier = forward_output.scale_modifier
        gsgn_data_spec.viewmatrix = forward_output.viewmatrices
        gsgn_data_spec.projmatrix = forward_output.projmatrices
        gsgn_data_spec.tan_fovx = forward_output.tanfovxs
        gsgn_data_spec.tan_fovy = forward_output.tanfovys
        gsgn_data_spec.cx = forward_output.cxs
        gsgn_data_spec.cy = forward_output.cys
        gsgn_data_spec.sh = shs.contiguous() if shs is not None else torch.empty(0, device="cuda")
        gsgn_data_spec.unactivated_opacity = unactivated_opacities.contiguous() if unactivated_opacities is not None else torch.empty(0, device="cuda")
        gsgn_data_spec.unactivated_scales = unactivated_scales.contiguous() if unactivated_scales is not None else torch.empty(0, device="cuda")
        gsgn_data_spec.unactivated_rotations = unactivated_rotations.contiguous() if unactivated_rotations is not None else torch.empty(0, device="cuda")
        gsgn_data_spec.degree = forward_output.sh_degree
        gsgn_data_spec.H = forward_output.H
        gsgn_data_spec.W = forward_output.W
        gsgn_data_spec.campos = forward_output.camposes
        gsgn_data_spec.R = forward_output.num_rendered_list
        gsgn_data_spec.binningBuffer = forward_output.binningBuffers
        gsgn_data_spec.imageBuffer = forward_output.imgBuffers
        gsgn_data_spec.use_double_precision = use_double_precision
        gsgn_data_spec.debug = forward_output.debug

        dummy = torch.empty([0], device=params.device, dtype=params.dtype)
        gsgn_data_spec.residuals = forward_output.residuals if forward_output.residuals is not None else dummy
        gsgn_data_spec.weights = forward_output.weights if forward_output.weights is not None else dummy
        gsgn_data_spec.residuals_ssim = forward_output.residuals_ssim if forward_output.residuals_ssim is not None else dummy
        gsgn_data_spec.weights_ssim = forward_output.weights_ssim if forward_output.weights_ssim is not None else dummy

        if forward_output.n_contrib_vol_rend:
            x = torch.stack(forward_output.n_contrib_vol_rend, dim=0)
            num_images, H, W = x.shape
            def pad(t, warp_size, dim):
                pad_size = t.shape[dim] % warp_size
                if pad_size > 0:
                    padding = torch.zeros(list(t.shape[:dim]) + [warp_size - pad_size] + list(t.shape[dim+1:]), device=t.device, dtype=t.dtype)
                    return torch.cat([t, padding], dim=dim)
                return t
            x_padded = pad(pad(x, 16, 2), 2, 1)
            x_sum = x_padded.unfold(1, 2, 2).unfold(2, 16, 16).sum(dim=(-1, -2))
            x_flat = x_sum.view(num_images, -1)
            x_cumsum = torch.cumsum(x_flat, dim=1)
            num_sparse_gaussians = [item.item() for item in x_cumsum[:, -1]]
            x_prefix = x_cumsum[:, :-1]
            zeros = torch.zeros_like(x_prefix[:, 0:1])
            x_final = torch.cat([zeros, x_prefix], dim=1)
            gsgn_data_spec.n_contrib_vol_rend_prefix_sum = [row for row in x_final]
            gsgn_data_spec.num_sparse_gaussians = num_sparse_gaussians
        else:
            gsgn_data_spec.n_contrib_vol_rend_prefix_sum = []
            gsgn_data_spec.num_sparse_gaussians = [0] * forward_output.num_images

        if forward_output.is_gaussian_hit:
            valid_mask = torch.stack(forward_output.is_gaussian_hit, dim=0)
            x = torch.cumsum(valid_mask, dim=1, dtype=torch.int32)
            num_visible_gaussians = [item.item() for item in x[:, -1]]
            gsgn_data_spec.map_visible_gaussians = [row for row in (x - 1)]
            gsgn_data_spec.num_visible_gaussians = num_visible_gaussians
            # Mocking map_cache_to_gaussians for demo
            gsgn_data_spec.map_cache_to_gaussians = [torch.arange(n, device="cuda", dtype=torch.int32) for n in num_visible_gaussians]
        else:
            gsgn_data_spec.map_visible_gaussians = []
            gsgn_data_spec.num_visible_gaussians = [0] * forward_output.num_images
            gsgn_data_spec.map_cache_to_gaussians = []

        gsgn_data_spec.geomBuffer = forward_output.geomBuffers
        
        return gsgn_data_spec
    
    @staticmethod
    def eval_jtf_RHS_only(
        params: torch.Tensor,
        means3D: torch.Tensor = None,
        shs: torch.Tensor = None,
        unactivated_opacities: torch.Tensor = None,
        unactivated_scales: torch.Tensor = None,
        unactivated_rotations: torch.Tensor = None,
        forward_output: RenderedImageAndBackwardValues = None,
        use_double_precision: bool = False,
        timing_dict: Dict[str, float] = None):

        if forward_output is None: raise Exception('Please provide forward output!')

        with measure_time("eval_jtf_RHS_only_python", timing_dict, maximum=True):
            data = GaussianRasterizer.build_gsgn_data_spec(
                params=params, means3D=means3D, shs=shs, 
                unactivated_opacities=unactivated_opacities, 
                unactivated_scales=unactivated_scales, 
                unactivated_rotations=unactivated_rotations,
                forward_output=forward_output, 
                use_double_precision=use_double_precision
            )
            
            if forward_output.weights is not None: data.residuals = data.residuals * forward_output.weights
            if forward_output.residuals_ssim is not None and forward_output.weights_ssim is not None:
                data.residuals += data.residuals_ssim * forward_output.weights_ssim

            # Invoke C++/CUDA rasterizer
            r, sparse_jacobians, index_maps, per_gaussian_caches = safe_call_fn(_C.eval_jtf_RHS_only, [data], forward_output.debug)

            # Free C++ GSGNDataSpec's internal pointers
            data.free_pointer_memory()

        return r, sparse_jacobians, index_maps, per_gaussian_caches, data
