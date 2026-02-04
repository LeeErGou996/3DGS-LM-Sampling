#pragma once

#include <torch/extension.h>
#include <vector>
#include <string>
#define GLM_FORCE_CUDA
#include <glm/glm.hpp>

// Forward declarations for CUDA types
struct float3;
struct float4;
struct uint2;
typedef unsigned short __half;


// From config.h mock
#define GSGN_NUM_CHANNELS 3
#define CHECK_CUDA(a,b) // Mock CHECK_CUDA

// From submodules/diff-gaussian-rasterization/cuda_rasterizer/gsgn_data_spec.h
#define TORCH_CHECK_CUDA(x) TORCH_CHECK(x.is_cuda(), #x " must be a CUDA tensor")
#define TORCH_CHECK_CONTIGUOUS(x) TORCH_CHECK(x.is_contiguous(), #x " must be contiguous")
#define CHECK_INPUT(x) TORCH_CHECK_CUDA(x); TORCH_CHECK_CONTIGUOUS(x)

using torch::Tensor;

struct GSGNDataSpec {
    // inputs provided all the time
    Tensor background;
    Tensor params;
	Tensor means3D;
	float scale_modifier;
	std::vector<Tensor> viewmatrix;
    std::vector<Tensor> projmatrix;
	Tensor tan_fovx;
	Tensor tan_fovy;
    Tensor cx;
    Tensor cy;
	Tensor sh;
    Tensor unactivated_opacity;
	Tensor unactivated_scales;
	Tensor unactivated_rotations;
	int degree; // degree of SH coeffs (0, 1, 2, 3)
    int H; // image height
    int W; // image width
	Tensor campos;
	std::vector<Tensor> geomBuffer;
	std::vector<int> R;  // num_rendered_list
	std::vector<Tensor> binningBuffer;
	std::vector<Tensor> imageBuffer;
    std::vector<int> num_sparse_gaussians;
    std::vector<Tensor> map_visible_gaussians;
    std::vector<Tensor> map_cache_to_gaussians;
    std::vector<int> num_visible_gaussians;
	bool debug;
    bool use_double_precision;

    // inputs that can be optional (not used by every GSGN function)
    bool have_n_contrib_vol_rend_prefix_sum = false;
    std::vector<Tensor> n_contrib_vol_rend_prefix_sum;
    bool have_residuals = false;
    Tensor residuals;
    bool have_weights = false;
    Tensor weights;
    bool have_residuals_ssim = false;
    Tensor residuals_ssim;
    bool have_weights_ssim = false;
    Tensor weights_ssim;

    // pointers created for later usage
    int* num_rendered_ptrs = nullptr;
    float** viewmatrix_ptrs = nullptr;
    float** projmatrix_ptrs = nullptr;
    char** geomBuffer_ptrs = nullptr;
    char** binningBuffer_ptrs = nullptr;
    char** imageBuffer_ptrs = nullptr;
    int** n_contrib_vol_rend_prefix_sum_ptrs = nullptr;
    int* n_sparse_gaussians = nullptr;
    int32_t max_n_sparse_gaussians = 0;
    int** map_visible_gaussians_ptrs = nullptr;
    int** map_cache_to_gaussians_ptrs = nullptr;
    int* n_visible_gaussians = nullptr;
    int32_t max_n_visible_gaussians = 0;

    // extracted values
    int P = 0; // number of gaussians
    uint32_t num_images = 0; // number of images
    int num_pixels = 0; // number of pixels in each image
    int jx_stride = 0;
    int M = 0; // total number of SH coeffs per channel (1, 4, 9, 16)
    Tensor focal_x, focal_y; // calculated from tan_fovx and tan_fovy

    // param offset values for input/output vectors
    int64_t total_params = 0;
    int64_t offset_xyz = 0, offset_scales = 0, offset_rotations = 0, offset_opacity = 0, offset_features_dc = 0, offset_features_rest = 0;

    inline void check();
    inline void allocate_pointer_memory();
    inline void free_pointer_memory();
    inline void get_parameter_offsets();
    inline void init();
};

namespace CudaRasterizer
{
    // Mocking GeometryStateReduced
    struct GeometryStateReduced {
        bool clamped[GSGN_NUM_CHANNELS];
        bool radius_gt_zero;
        float means2D[2];
        float cov3D[6];
        float conic_opacity[4];
        float rgb[GSGN_NUM_CHANNELS];
    };

    // Mocking ImageState
    struct ImageState {
        uint2* ranges;
        int32_t* n_contrib;
        float* accum_alpha;
    };

    // Mocking BinningStateReduced
    struct BinningStateReduced {
        uint32_t* point_list;
    };

    struct PackedGSGNDataSpec {
        PackedGSGNDataSpec(GSGNDataSpec& data);

        // inputs from GSGNDataSpec
        int P;
        int D;
        int M;
        int* num_rendered;
        float* __restrict__ background;
        int W;
        int H;
        int num_pixels;
        int jx_stride;
        int num_images;
        float* __restrict__ params;
        float3* __restrict__ means3D;
        float* __restrict__ shs;
        float* __restrict__ unactivated_opacity;
        float3* __restrict__ unactivated_scales;
        float4* __restrict__ unactivated_rotations;
        float scale_modifier;
        float** viewmatrix;
        float** projmatrix;
        glm::vec3* __restrict__ campos;
        float* tan_fovx;
        float* tan_fovy;
        float* cx;
        float* cy;
        float* focal_x;
        float* focal_y;
        bool have_n_contrib_vol_rend_prefix_sum;
        int** n_contrib_vol_rend_prefix_sum;
        bool have_residuals;
        float* residuals;
        bool have_weights;
        float* weights;
        bool have_residuals_ssim;
        float* residuals_ssim;
        bool have_weights_ssim;
        float* weights_ssim;
        int** map_visible_gaussians;
        int** map_cache_to_gaussians;
        int* n_visible_gaussians;
        bool debug;
        const int64_t offset_xyz, offset_scales, offset_rotations, offset_opacity, offset_features_dc, offset_features_rest;
        int* n_sparse_gaussians;

        // inputs created during construction
        int32_t max_n_sparse_gaussians;
        int32_t max_n_visible_gaussians;

        // pointers created for later usage
        bool pointers_changed = false;
        char** geomBuffer_ptrs;
        char** binningBuffer_ptrs;
        char** imageBuffer_ptrs;
        uint2** ranges_ptrs;
        int32_t** n_contrib_ptrs;
        float** accum_alpha_ptrs;
        uint32_t** point_list_ptrs;

        inline void allocate_pointer_memory();
        inline void free_pointer_memory();
    };
}

// Declaration of the function to be bound
std::tuple<torch::Tensor, std::vector<torch::Tensor>, std::vector<torch::Tensor>, std::vector<torch::Tensor>>
EvalJTFRHSOnly(GSGNDataSpec& data);
