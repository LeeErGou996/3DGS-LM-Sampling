import torch
import os
import sys
import random
import math
from typing import Dict, List
from contextlib import contextmanager

# Add the current directory to Python path to import custom modules
sys.path.append(os.path.dirname(os.path.abspath(__file__)))

# Import the modified Python functions
try:
    from python_functions import GaussianRasterizer, RenderedImageAndBackwardValues
    import rhs_cuda_extension._C_RHS as _C
except ImportError as e:
    print(f"Failed to import a necessary module: {e}")
    print("Please make sure you have compiled the C++ extension by running 'python setup.py install'")
    sys.exit(1)

# ========================================================================================
# Mock classes and functions for the demo
# ========================================================================================

class MockGaussianModel:
    def __init__(self, num_gaussians=10, sh_degree=0):
        self._num_gaussians = num_gaussians
        self._sh_degree = sh_degree
        self._num_sh = (sh_degree + 1)**2
        self._num_sh_rest = self._num_sh - 1

        # Initialize tensors on CUDA
        self._xyz = torch.randn(self._num_gaussians, 3, device="cuda")
        self._rotation = torch.randn(self._num_gaussians, 4, device="cuda")
        self._scaling = torch.randn(self._num_gaussians, 3, device="cuda")
        self._opacity = torch.randn(self._num_gaussians, 1, device="cuda")
        self._features_dc = torch.randn(self._num_gaussians, 1, 3, device="cuda")
        self._features_rest = torch.randn(self._num_gaussians, self._num_sh_rest, 3, device="cuda")
    
    @property
    def get_xyz(self): return self._xyz
    @property
    def get_rotation(self): return self._rotation
    @property
    def get_scaling(self): return self._scaling
    @property
    def get_opacity(self): return self._opacity
    @property
    def get_active_features_rest(self): return self._features_rest

    def get_reordered_params(self, with_SH=False):
        xyz_flat = self._xyz.T.reshape(-1)
        scales_flat = self._scaling.T.reshape(-1)
        rotations_flat = self._rotation.T.reshape(-1)
        opacity_flat = self._opacity.reshape(-1)
        sh_dc_flat = self._features_dc.permute(2, 0, 1).reshape(-1)
        sh_rest_flat = self._features_rest.permute(2, 1, 0).reshape(-1)
        return torch.cat([xyz_flat, scales_flat, rotations_flat, opacity_flat, sh_dc_flat, sh_rest_flat])

    def get_resorted_vec(self, x): return x

class MockCamera:
    def __init__(self, image_width=128, image_height=128):
        self.image_width = image_width
        self.image_height = image_height

class MockOptimizationParams: pass
class MockPipelineParams:
    def __init__(self, debug=False): self.debug = debug

class MockGSGNParams:
    def __init__(self):
        self.enable_ssgn = True
        self.image_subsample_size = 1
        self.ssgn_size_rhs = 1
        self.ssgn_size_lhs = 1
        self.use_double_precision = False
        self.compute_huber_weights = False
        self.huber_c = 0.0
        self.compute_ssim_weights = False
        self.ssim_residual_scale_factor = 0.0
        self.min_lm_diagonal = 1e-6
        self.max_lm_diagonal = 1e6
        self.max_grad_norm = float('inf')
        self.pcg_max_iter = 10
        self.pcg_atol = 1e-6
        self.pcg_rtol = 1e-6
        self.pcg_gradient_descent_every = 0
        self.pcg_explicit_residual_every = 0
        self.pcg_verbose = False
        self.image_subsample_frame_selection_mode = "random"


@contextmanager
def measure_time_mock(name: str, out_dict: Dict[str, float], additive: bool = False, maximum: bool = False):
    yield

def get_subsample_indices(indices: List[int], n_images: int, gsgn_params: MockGSGNParams) -> List[int]:
    if gsgn_params.image_subsample_size == 0: return []
    random.shuffle(indices)
    return indices[:gsgn_params.image_subsample_size]

def render_all_images_and_backward(gaussians, viewpoint_stack, opt, pipe, background, prepare_for_gsgn_backward, compute_huber_weights, huber_c, compute_ssim_weights, ssim_residual_scale_factor):
    P = gaussians._num_gaussians
    num_images = len(viewpoint_stack)
    H, W = viewpoint_stack[0].image_height, viewpoint_stack[0].image_width
    GSGN_NUM_CHANNELS = 3
    
    forward_output = RenderedImageAndBackwardValues(
        num_images=num_images, H=H, W=W, bg=background,
        scale_modifier=1.0, sh_degree=gaussians._sh_degree, debug=pipe.debug,
        residuals=torch.randn(GSGN_NUM_CHANNELS * num_images * H * W, device="cuda"),
    )
    for _ in range(num_images):
        forward_output.viewmatrices.append(torch.eye(4, device="cuda"))
        forward_output.projmatrices.append(torch.eye(4, device="cuda")),
        forward_output.camposes.append(torch.zeros(3, device="cuda")),
        forward_output.tanfovxs.append(0.5)
        forward_output.tanfovys.append(0.5)
        forward_output.cxs.append(W/2)
        forward_output.cys.append(H/2)
        forward_output.geomBuffers.append(torch.empty(P * 64, device="cuda", dtype=torch.uint8))
        forward_output.binningBuffers.append(torch.empty(100, device="cuda", dtype=torch.uint8))
        forward_output.imgBuffers.append(torch.empty(100, device="cuda", dtype=torch.uint8))
        forward_output.num_rendered_list.append(100)
        forward_output.n_contrib_vol_rend.append(torch.randint(1, 5, (H, W), dtype=torch.int32, device="cuda")),
        forward_output.is_gaussian_hit.append(torch.ones(P, dtype=torch.bool, device="cuda")),
    
    forward_output.tanfovxs = torch.tensor(forward_output.tanfovxs, device="cuda", dtype=torch.float32)
    forward_output.tanfovys = torch.tensor(forward_output.tanfovys, device="cuda", dtype=torch.float32)
    forward_output.cxs = torch.tensor(forward_output.cxs, device="cuda", dtype=torch.float32)
    forward_output.cys = torch.tensor(forward_output.cys, device="cuda", dtype=torch.float32)
    forward_output.camposes = torch.stack(forward_output.camposes)
    return forward_output

def eval_jtf_RHS_only(gaussians, forward_output, timing_dict, use_double_precision):
    return GaussianRasterizer.eval_jtf_RHS_only(
        params=gaussians.get_reordered_params(with_SH=False),
        means3D=gaussians.get_xyz,
        shs=gaussians.get_active_features_rest,
        unactivated_opacities=gaussians.get_opacity,
        unactivated_scales=gaussians.get_scaling,
        unactivated_rotations=gaussians.get_rotation,
        forward_output=forward_output,
        use_double_precision=use_double_precision,
        timing_dict=timing_dict
    )

def run_ssgn_verification_demo():
    print("--- Starting SSGN Verification Demo ---")

    NUM_GAUSSIANS = 100
    NUM_IMAGES = 5
    IMAGE_WIDTH = 128
    IMAGE_HEIGHT = 128
    SH_DEGREE = 1
    RHS_MICRO_BATCH_SIZE = 2 # Set to > 1 to test accumulation
    LHS_BATCH_SIZE = 1

    gaussians = MockGaussianModel(num_gaussians=NUM_GAUSSIANS, sh_degree=SH_DEGREE)
    viewpoint_stack = [MockCamera(image_width=IMAGE_WIDTH, image_height=IMAGE_HEIGHT) for _ in range(NUM_IMAGES)]
    viewpoint_stack_indices = list(range(NUM_IMAGES))
    
    opt = MockOptimizationParams()
    pipe = MockPipelineParams(debug=True)
    background = torch.zeros(3, device="cuda")

    gsgn = MockGSGNParams()
    gsgn.ssgn_size_rhs = RHS_MICRO_BATCH_SIZE
    gsgn.ssgn_size_lhs = LHS_BATCH_SIZE
    
    timing_dict = {}

    try:
        print(f"Running SSGN RHS test with {NUM_GAUSSIANS} gaussians and {NUM_IMAGES} images.")
        print(f"RHS Micro Batch Size: {RHS_MICRO_BATCH_SIZE}")

        # This logic is adapted from linear_solve_pcg_fused to isolate the RHS part
        original_size = gsgn.image_subsample_size
        gsgn.image_subsample_size = gsgn.ssgn_size_rhs
        
        indices_rhs = get_subsample_indices(viewpoint_stack_indices.copy(), len(viewpoint_stack), gsgn)
        images_rhs = [viewpoint_stack[i] for i in indices_rhs]
        
        with measure_time_mock("render_rhs", timing_dict, additive=True):
            out_rhs = render_all_images_and_backward(
                gaussians=gaussians, viewpoint_stack=images_rhs, opt=opt, pipe=pipe, background=background,
                prepare_for_gsgn_backward=True, compute_huber_weights=False, huber_c=0,
                compute_ssim_weights=False, ssim_residual_scale_factor=0
            )
        
        print("Calling C++/CUDA eval_jtf_RHS_only...")
        b, sparse_jacobians, index_maps, per_gaussian_caches, data = eval_jtf_RHS_only(
            gaussians, forward_output=out_rhs,
            timing_dict=timing_dict, use_double_precision=gsgn.use_double_precision
        )

        print("\n--- eval_jtf_RHS_only executed successfully! ---")
        print("b (RHS vector) shape:", b.shape)
        assert b.shape[0] == gaussians.get_reordered_params(False).shape[0], "Shape of 'b' vector is incorrect!"
        
        print(f"Num sparse_jacobians lists: {len(sparse_jacobians)}")
        print(f"Num index_maps lists: {len(index_maps)}")
        print(f"Num per_gaussian_caches lists: {len(per_gaussian_caches)}")

        print("\nDemo finished successfully. The C++/CUDA function for RHS calculation was called and ran without crashing.")

    except Exception as e:
        import traceback
        print(f"\n--- Demo failed with an error: {e} ---")
        traceback.print_exc()
        print("\nThis could indicate a bug in the C++/CUDA code, a binding issue, or incorrect mock data.")

    print("\n--- SSGN Verification Demo End ---")

if __name__ == "__main__":
    if not torch.cuda.is_available():
        print("CUDA is not available. This demo requires a CUDA-enabled GPU.")
    else:
        run_ssgn_verification_demo()