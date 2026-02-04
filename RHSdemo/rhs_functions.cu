#include "rhs_functions.h"

// CUDA specific includes and mocks
#include <cuda_runtime.h>
#include <ATen/ATen.h>
#include <cooperative_groups.h>

// Mocking glm and cuda types for compilation
struct __align__(4) float2 { float x, y; };
struct __align__(8) float3 { float x, y, z; };
struct __align__(16) float4 { float x, y, z, w; };
struct __align__(8) uint2 { unsigned int x, y; };

// Mock atomics
template<typename T> __device__ void atomicAdd(T* address, T val) {}
template<typename T> __device__ T dsigmoidvdv(T x, T y) { return x * y; } // Mocked function
template<typename T> __device__ T dexpvdv(T x, T y) { return x * y; } // Mocked function
template<typename T> __device__ T dnormvdv(float4 x, float4 y) { return y.x; } // Mocked function for float4
__device__ float3 dnormvdv(float x, float y, float z, float dx, float dy, float dz) { return {dx, dy, dz}; } // Mocked function for float3


// Mock cooperative_groups
namespace cooperative_groups {
    struct thread_block {
        __device__ uint32_t group_index() { return 0; }
        __device__ uint32_t thread_rank() { return 0; }
        __device__ uint32_t thread_index() { return 0; }
    };
    __device__ thread_block this_thread_block() { return {}; }
    struct grid_group {
        __device__ uint32_t thread_rank() { return 0; }
    };
    __device__ grid_group this_grid() { return {}; }

    template<typename T> struct WarpScan {
        struct TempStorage {};
        __device__ WarpScan(TempStorage&) {}
        __device__ void ExclusiveSum(T, T&, T&) {}
        __device__ T HeadSegmentedSum(T val, bool head_flag) { return val; }
    };
}
namespace cub { // mock cub::WarpScan
    template<typename T> struct WarpScan {
        struct TempStorage {};
        __device__ WarpScan(TempStorage&) {}
        __device__ void ExclusiveSum(T, T&, T&) {}
        __device__ T HeadSegmentedSum(T val, bool head_flag) { return val; }
    };
}

// Mock macros
#define __forceinline__ 
#define __global__ 
#define __device__ 
#define __host__ 
#define __launch_bounds__(x) 
#define __restrict__
#define __syncwarp()
#define __shfl_up_sync(mask, val, delta) val
#define __ffs(mask) 0
#define __popc(mask) 0
#define __activemask() 0xFFFFFFFF
#define __half2float(x) (float)x
#define __float2half(x) (unsigned short)x
#define __expf(x) expf(x)


// Fakedefinitions to allow compilation
#define GSGN_BLOCK_X 16
#define GSGN_BLOCK_Y 16
#define GSGN_ALPHA_THRESH 1.f / 255.f


// Mock structures for CUDA kernels
struct GradientCache {
    __half dchannel_dcolor;
    __half dL_dalpha[GSGN_NUM_CHANNELS];
};
struct GaussianCache {
    float T[6];
    float a_;
    float b_;
    float c_;
    float x_grad_mul;
    float y_grad_mul;
    float dL_dT_precomp[6];
    float denom;
    float denom2inv;
    float dRGBdx[GSGN_NUM_CHANNELS];
    float dRGBdy[GSGN_NUM_CHANNELS];
    float dRGBdz[GSGN_NUM_CHANNELS];
    float R[9];
    float t[4];
};
struct GaussianAttributeNoSH {
    float mean3D[3];
    float unactivated_scale[3];
    float unactivated_rotation[4];
    float unactivated_opacity;
};
struct GaussianCacheComputeCov2D { float T[6]; };
struct GaussianCachePreprocess { float R[9]; };
enum class GSGN_MODE { EVAL_JTF_AND_SPARSE_INTERMEDIATE, APPLY_JTJ, PRECONDITIONER, APPLY_J };


// Mock `get_vector_position`
template <typename T> __device__ int32_t get_vector_position(const int32_t global_id, const int32_t channel, const uint32_t idx, CudaRasterizer::PackedGSGNDataSpec& data) {
    return global_id; // Simplified mock
}
// Mock `atomred_vec`
template<typename ATOM_T> __device__ void atomred_vec(unsigned int laneId, size_t idx, ATOM_T** ptr, ATOM_T *val, size_t len, unsigned int balance_threshold) {}

// Mock `transformPoint4x3`
__device__ float4 transformPoint4x4(float3 p, const float* mat) { return {p.x, p.y, p.z, 1.0f}; }
// Mock `transformVec4x3Transpose`
__device__ glm::vec3 transformVec4x3Transpose(float dx, float dy, float dz, const float* mat) { return {dx, dy, dz}; }
__device__ float3 transformVec4x3Transpose(float3 vec, const float* mat) { return vec; }


// Mock kernel for gsgn_computeCov2DCUDA
template <typename scalar_t, bool write_cache>
__global__ void __launch_bounds__(256)
gsgn_computeCov2DCUDA(
    CudaRasterizer::PackedGSGNDataSpec data,
    const int img_id,
    const int n_visible_gaussians,
    const scalar_t* dL_dconics,
    scalar_t* out_vec,
    scalar_t* dL_dcov,
    float* per_gaussian_cache,
    int* map_cache_to_gaussians) {
    // Mock kernel implementation
}

// Mock kernel for gsgn_preprocessCUDA
template <typename scalar_t, bool write_cache>
__global__ void __launch_bounds__(256)
gsgn_preprocessCUDA(
    CudaRasterizer::PackedGSGNDataSpec data,
    const int img_id,
    const int n_visible_gaussians,
    const scalar_t* dL_dmean2D,
    const scalar_t* dL_dcolor,
    const scalar_t* dL_dcov3D,
    scalar_t* out_vec,
    float* per_gaussian_cache,
    int* map_cache_to_gaussians) {
    // Mock kernel implementation
}


//================================================================================
// Implementations of inline methods for GSGNDataSpec
//================================================================================
inline void GSGNDataSpec::check() {
    // Mocking check for demo purpose
    P = means3D.size(0);
    num_images = geomBuffer.size();
    num_pixels = H * W;
    if(tan_fovx.numel() > 0) focal_x = W / (2.0f * tan_fovx.mean());
    if(tan_fovy.numel() > 0) focal_y = H / (2.0f * tan_fovy.mean()); 
    if(sh.numel() > 0) M = sh.size(1) + 1;

    jx_stride = num_images * W * H;

    have_n_contrib_vol_rend_prefix_sum = n_contrib_vol_rend_prefix_sum.size() == num_images;
    have_residuals = residuals.numel() > 0;
    have_weights = weights.numel() > 0;
    have_residuals_ssim = residuals_ssim.numel() > 0;
    have_weights_ssim = weights_ssim.numel() > 0;
}

inline void GSGNDataSpec::allocate_pointer_memory() {
    cudaMallocManaged((void**) &num_rendered_ptrs, num_images * sizeof(int));
    cudaMallocManaged((void**) &viewmatrix_ptrs, num_images * sizeof(float*));
    cudaMallocManaged((void**) &projmatrix_ptrs, num_images * sizeof(float*));
    cudaMallocManaged((void**) &geomBuffer_ptrs, num_images * sizeof(char*));
    cudaMallocManaged((void**) &binningBuffer_ptrs, num_images * sizeof(char*));
    cudaMallocManaged((void**) &imageBuffer_ptrs, num_images * sizeof(char*));
    cudaMallocManaged((void**) &map_visible_gaussians_ptrs, num_images * sizeof(int*));
    cudaMallocManaged((void**) &map_cache_to_gaussians_ptrs, num_images * sizeof(int*));
    cudaMallocManaged((void**) &n_visible_gaussians, num_images * sizeof(int));
    cudaMallocManaged((void**) &n_sparse_gaussians, num_images * sizeof(int));
    if(have_n_contrib_vol_rend_prefix_sum) {
        cudaMallocManaged((void**) &n_contrib_vol_rend_prefix_sum_ptrs, num_images * sizeof(int*));
    }

    for(uint32_t i=0; i < num_images; i++) {
        num_rendered_ptrs[i] = R[i];
        viewmatrix_ptrs[i] = viewmatrix[i].data_ptr<float>();
        projmatrix_ptrs[i] = projmatrix[i].data_ptr<float>();
        geomBuffer_ptrs[i] = reinterpret_cast<char*>(geomBuffer[i].data_ptr());
        binningBuffer_ptrs[i] = reinterpret_cast<char*>(binningBuffer[i].data_ptr());
        imageBuffer_ptrs[i] = reinterpret_cast<char*>(imageBuffer[i].data_ptr());
        map_visible_gaussians_ptrs[i] = map_visible_gaussians[i].data_ptr<int>();
        map_cache_to_gaussians_ptrs[i] = map_cache_to_gaussians[i].data_ptr<int>();
        int x_visible = this->num_visible_gaussians[i];
        n_visible_gaussians[i] = x_visible;
        max_n_visible_gaussians = x_visible > max_n_visible_gaussians ? x_visible : max_n_visible_gaussians;
        int x_sparse = this->num_sparse_gaussians[i];
        n_sparse_gaussians[i] = x_sparse;
        max_n_sparse_gaussians = x_sparse > max_n_sparse_gaussians ? x_sparse : max_n_sparse_gaussians;
        if(have_n_contrib_vol_rend_prefix_sum) {
            n_contrib_vol_rend_prefix_sum_ptrs[i] = n_contrib_vol_rend_prefix_sum[i].data_ptr<int>();
        }
    }
}

inline void GSGNDataSpec::free_pointer_memory() {
    cudaDeviceSynchronize();
    cudaFree(num_rendered_ptrs);
    cudaFree(viewmatrix_ptrs);
    cudaFree(projmatrix_ptrs);
    cudaFree(geomBuffer_ptrs);
    cudaFree(binningBuffer_ptrs);
    cudaFree(imageBuffer_ptrs);
    cudaFree(n_sparse_gaussians);
    cudaFree(map_visible_gaussians_ptrs);
    cudaFree(map_cache_to_gaussians_ptrs);
    cudaFree(n_visible_gaussians);
    if(have_n_contrib_vol_rend_prefix_sum) {
        cudaFree(n_contrib_vol_rend_prefix_sum_ptrs);
    }
}

inline void GSGNDataSpec::get_parameter_offsets() {
    int64_t params_xyz = P * 3;
    offset_xyz = 0;

    int64_t params_scales = P * 3;
    offset_scales = offset_xyz + params_xyz;

    int64_t params_rotations = P * 4;
    offset_rotations = offset_scales + params_scales;

    int64_t params_opacity = P;
    offset_opacity = offset_rotations + params_rotations;

    int64_t params_features_dc = P * GSGN_NUM_CHANNELS;
    offset_features_dc = offset_opacity + params_opacity;

    int32_t params_per_channel_rest = (M - 1);
    int64_t params_features_rest = P * params_per_channel_rest * GSGN_NUM_CHANNELS;
    offset_features_rest = offset_features_dc + params_features_dc;

    total_params = params_xyz + params_scales + params_rotations + params_opacity + params_features_dc + params_features_rest;
}

inline void GSGNDataSpec::init() {
    check();
    allocate_pointer_memory();
    get_parameter_offsets();
}

CudaRasterizer::PackedGSGNDataSpec::PackedGSGNDataSpec(GSGNDataSpec& data):
    P(data.P),
    D(data.degree),
    M(data.M),
    num_rendered(data.num_rendered_ptrs),
    background(data.background.data_ptr<float>()),
    W(data.W),
    H(data.H),
    num_pixels(data.num_pixels),
    jx_stride(data.jx_stride),
    num_images(data.num_images),
    params(data.params.data_ptr<float>()),
    means3D((float3*)data.means3D.data_ptr<float>()),
    shs(data.sh.data_ptr<float>()),
    unactivated_opacity(data.unactivated_opacity.data_ptr<float>()),
    unactivated_scales((float3*)data.unactivated_scales.data_ptr<float>()),
    unactivated_rotations((float4*)data.unactivated_rotations.data_ptr<float>()),
    scale_modifier(data.scale_modifier),
    viewmatrix(data.viewmatrix_ptrs),
    projmatrix(data.projmatrix_ptrs),
    campos((glm::vec3*)data.campos.data_ptr<float>()),
    tan_fovx(data.tan_fovx.data_ptr<float>()),
    tan_fovy(data.tan_fovy.data_ptr<float>()),
    cx(data.cx.data_ptr<float>()),
    cy(data.cy.data_ptr<float>()),
    focal_x(data.focal_x.data_ptr<float>()),
    focal_y(data.focal_y.data_ptr<float>()),
    have_n_contrib_vol_rend_prefix_sum(data.have_n_contrib_vol_rend_prefix_sum),
    n_contrib_vol_rend_prefix_sum(data.n_contrib_vol_rend_prefix_sum_ptrs),
    have_residuals(data.have_residuals),
    residuals(data.residuals.data_ptr<float>()),
    have_weights(data.have_weights),
    weights(data.weights.data_ptr<float>()),
    have_residuals_ssim(data.have_residuals_ssim),
    residuals_ssim(data.residuals_ssim.data_ptr<float>()),
    have_weights_ssim(data.have_weights_ssim),
    weights_ssim(data.weights_ssim.data_ptr<float>()),
    map_visible_gaussians(data.map_visible_gaussians_ptrs),
    map_cache_to_gaussians(data.map_cache_to_gaussians_ptrs),
    n_visible_gaussians(data.n_visible_gaussians),
    debug(data.debug),
    offset_xyz(data.offset_xyz),
    offset_scales(data.offset_scales),
    offset_rotations(data.offset_rotations),
    offset_opacity(data.offset_opacity),
    offset_features_dc(data.offset_features_dc),
    offset_features_rest(data.offset_features_rest),
    n_sparse_gaussians(data.n_sparse_gaussians) {

    max_n_sparse_gaussians = data.max_n_sparse_gaussians;
    max_n_visible_gaussians = data.max_n_visible_gaussians;

    cudaMallocManaged((void**) &geomBuffer_ptrs, num_images * sizeof(char*));
    cudaMallocManaged((void**) &binningBuffer_ptrs, num_images * sizeof(char*));
    cudaMallocManaged((void**) &imageBuffer_ptrs, num_images * sizeof(char*));
    for(int i = 0; i < num_images; i++) {
        geomBuffer_ptrs[i] = data.geomBuffer_ptrs[i];
        binningBuffer_ptrs[i] = data.binningBuffer_ptrs[i];
        imageBuffer_ptrs[i] = data.imageBuffer_ptrs[i];
    }
}

inline void CudaRasterizer::PackedGSGNDataSpec::allocate_pointer_memory() {
    cudaMallocManaged((void**) &ranges_ptrs, num_images * sizeof(uint2*));
    cudaMallocManaged((void**) &n_contrib_ptrs, num_images * sizeof(int32_t*));
    cudaMallocManaged((void**) &accum_alpha_ptrs, num_images * sizeof(float*));
    cudaMallocManaged((void**) &point_list_ptrs, num_images * sizeof(uint32_t*));

    for(int i=0; i < num_images; i++) {
        // These would normally come from actual buffers. Mocking for demo.
        CudaRasterizer::ImageState imgState = { (uint2*)malloc(sizeof(uint2)), (int32_t*)malloc(sizeof(int32_t)), (float*)malloc(sizeof(float)) };
        ranges_ptrs[i] = imgState.ranges;
        n_contrib_ptrs[i] = imgState.n_contrib;
        accum_alpha_ptrs[i] = imgState.accum_alpha;
        
        CudaRasterizer::BinningStateReduced binningState = { (uint32_t*)malloc(sizeof(uint32_t)) };
        point_list_ptrs[i] = binningState.point_list;
    }
    pointers_changed = true;
}

inline void CudaRasterizer::PackedGSGNDataSpec::free_pointer_memory() {
    cudaDeviceSynchronize();
    cudaFree(geomBuffer_ptrs);
    cudaFree(binningBuffer_ptrs);
    cudaFree(imageBuffer_ptrs);
    if(pointers_changed) {
        for(int i=0; i<num_images; ++i) {
            free(ranges_ptrs[i]);
            free(n_contrib_ptrs[i]);
            free(accum_alpha_ptrs[i]);
            free(point_list_ptrs[i]);
        }
        cudaFree(ranges_ptrs);
        cudaFree(n_contrib_ptrs);
        cudaFree(accum_alpha_ptrs);
        cudaFree(point_list_ptrs);
    }
}

// -----------------------------------------------------------------------------------------
//  New SSGN Implementation: Forwarding to GSGN namespace
// -----------------------------------------------------------------------------------------

namespace CudaRasterizer {
    class Rasterizer {
    public:
        // Mocked declaration for eval_jtf_and_get_sparse_jacobian
        template<typename T> static void eval_jtf_and_get_sparse_jacobian(PackedGSGNDataSpec&, T*, __half**, int**, float**) {
             // In a real scenario, this would launch CUDA kernels
        }
        // Mocked declaration for eval_jtf_RHS_only
        template<typename T> static void eval_jtf_RHS_only(PackedGSGNDataSpec, T*, __half**, int**, float*);
    };
    
    namespace GSGN {
        template<typename T>
        __global__ void eval_jtf_RHS_only_kernel(
            PackedGSGNDataSpec data,
            const int img_id,
            const int dL_offset,
            T* __restrict__ r_vec,
            T* __restrict__ dL_dcolors,
            T* __restrict__ dL_dmean2D,
            T* __restrict__ dL_dconic2D,
            __half* __restrict__ sparse_jacobians,
            int* __restrict__ index_map);
            
        template<typename T> void eval_jtf_RHS_only(PackedGSGNDataSpec& data, T* r_vec, __half** sparse_jacobians, int** index_map, float** per_gaussian_cache);
    }
}


// 实现 Rasterizer 类的静态方法
template<typename T> void CudaRasterizer::Rasterizer::eval_jtf_RHS_only(CudaRasterizer::PackedGSGNDataSpec data, T* r_vec, __half** sparse_jacobians, int** index_map, float** per_gaussian_cache) {
    data.allocate_pointer_memory();
	CHECK_CUDA(CudaRasterizer::GSGN::eval_jtf_RHS_only<T>(data, r_vec, sparse_jacobians, index_map, per_gaussian_cache), data.debug)
    data.free_pointer_memory();
}
template void CudaRasterizer::Rasterizer::eval_jtf_RHS_only<float>(CudaRasterizer::PackedGSGNDataSpec data, float* r_vec, __half** sparse_jacobians, int** index_map, float** per_gaussian_cache);
template void CudaRasterizer::Rasterizer::eval_jtf_RHS_only<double>(CudaRasterizer::PackedGSGNDataSpec data, double* r_vec, __half** sparse_jacobians, int** index_map, float** per_gaussian_cache);


// -------------------------------------------------------------------------
// START: SSGN Optimization - RHS Only Kernel (Low VRAM)
// -------------------------------------------------------------------------

template <typename scalar_t>
__global__ void CudaRasterizer::GSGN::eval_jtf_RHS_only_kernel(
    CudaRasterizer::PackedGSGNDataSpec data,
    const int img_id,
    const int dL_offset,
    scalar_t* __restrict__ r_vec,
    scalar_t* __restrict__ dL_dcolors,
    scalar_t* __restrict__ dL_dmean2D,
    scalar_t* __restrict__ dL_dconic2D,
    __half* __restrict__ sparse_jacobians,
    int* __restrict__ index_map) {

    // Mock kernel implementation
}

// C++ Wrapper for the RHS-only kernel
template<typename T> void CudaRasterizer::GSGN::eval_jtf_RHS_only(CudaRasterizer::PackedGSGNDataSpec& data, T* r_vec, __half** sparse_jacobians, int** index_map, float** per_gaussian_cache) {
    T* helper_memory;
    cudaMallocManaged((void**) &helper_memory, data.max_n_visible_gaussians * (2 + 3 + GSGN_NUM_CHANNELS + 6) * sizeof(T));

    T* dL_dmeans2D = helper_memory;
    T* dL_dconic2D = dL_dmeans2D + data.max_n_visible_gaussians * 2;
    T* dL_dcolors = dL_dconic2D + data.max_n_visible_gaussians * 3;
    T* dL_dcov3D = dL_dcolors + data.max_n_visible_gaussians * GSGN_NUM_CHANNELS;

    dim3 block_rest = dim3(256, 1, 1);
    dim3 block_render = dim3(GSGN_BLOCK_X, GSGN_BLOCK_Y, 1);
    dim3 grid_render = dim3((data.W + block_render.x - 1) / block_render.x, (data.H + block_render.y - 1) / block_render.y, 1);

    for(int i = 0; i < data.num_images; i++) {
        int dL_offset = data.n_visible_gaussians[i];
        cudaMemset(helper_memory, 0, data.max_n_visible_gaussians * (2 + 3 + GSGN_NUM_CHANNELS) * sizeof(T));
        
        CudaRasterizer::GSGN::eval_jtf_RHS_only_kernel<T><<<grid_render, block_render>>>(
            data, i, dL_offset, r_vec, dL_dcolors, dL_dmeans2D, dL_dconic2D, sparse_jacobians[i], index_map[i]
        );

        dim3 grid_rest = dim3((dL_offset + 255) / 256, 1, 1);

        gsgn_computeCov2DCUDA<T, true><<<grid_rest, block_rest>>>(
            data, i, dL_offset, dL_dconic2D, r_vec, dL_dcov3D, per_gaussian_cache[i], data.map_cache_to_gaussians[i]
        );

        gsgn_preprocessCUDA<T, true><<<grid_rest, block_rest>>>(
            data, i, dL_offset, dL_dmeans2D, dL_dcolors, dL_dcov3D, r_vec, per_gaussian_cache[i], data.map_cache_to_gaussians[i]
        );
    }
    cudaFree(helper_memory);
}
template void CudaRasterizer::GSGN::eval_jtf_RHS_only<float>(CudaRasterizer::PackedGSGNDataSpec& data, float* r_vec, __half** sparse_jacobians, int** index_map, float** per_gaussian_cache);
template void CudaRasterizer::GSGN::eval_jtf_RHS_only<double>(CudaRasterizer::PackedGSGNDataSpec& data, double* r_vec, __half** sparse_jacobians, int** index_map, float** per_gaussian_cache);


std::tuple<torch::Tensor, std::vector<torch::Tensor>, std::vector<torch::Tensor>, std::vector<torch::Tensor>> EvalJTFRHSOnly(GSGNDataSpec& data) {
    data.init();

    auto options = data.params.options();
    if(data.use_double_precision) {
        options = options.dtype(torch::kFloat64);
    }
    torch::Tensor r_vec = torch::zeros({data.total_params}, options);

    std::vector<torch::Tensor> sparse_jacobians;
    std::vector<torch::Tensor> index_maps;
    std::vector<torch::Tensor> per_gaussian_caches;
    int** index_maps_ptr;
    __half** sparse_jacobians_ptr;
    float** per_gaussian_caches_ptr;

    cudaMallocManaged((void**) &index_maps_ptr, data.num_images * sizeof(int*));
    cudaMallocManaged((void**) &sparse_jacobians_ptr, data.num_images * sizeof(__half*));
    cudaMallocManaged((void**) &per_gaussian_caches_ptr, data.num_images * sizeof(float*));

    int per_gaussian_sparse_jac_size = 4;
    int per_gaussian_cache_size = 41;
    for(int i=0; i < data.num_images; i++) {
        int n_elem = data.num_sparse_gaussians[i];

        torch::Tensor sparse_jac = torch.empty({(long long)n_elem * per_gaussian_sparse_jac_size}, options.dtype(at::kHalf));
        sparse_jacobians.push_back(sparse_jac);
        sparse_jacobians_ptr[i] = (__half*) sparse_jac.contiguous().data_ptr<at::Half>();

        torch::Tensor index_map = torch.empty({(long long)n_elem * 2}, options.dtype(torch::kInt32));
        index_maps.push_back(index_map);
        index_maps_ptr[i] = index_map.contiguous().data_ptr<int>();

        torch::Tensor per_gaussian_cache = torch.empty({(long long)data.n_visible_gaussians[i] * per_gaussian_cache_size}, options.dtype(torch::kFloat32));
        per_gaussian_caches.push_back(per_gaussian_cache);
        per_gaussian_caches_ptr[i] = per_gaussian_cache.contiguous().data_ptr<float>();
    }

    if(data.P != 0) {
        CudaRasterizer::PackedGSGNDataSpec packed_data(data); // Pack data
        if(data.use_double_precision) {       
            CudaRasterizer::Rasterizer::eval_jtf_RHS_only<double>(
                packed_data,
                r_vec.contiguous().data_ptr<double>(),
                sparse_jacobians_ptr,
                index_maps_ptr,
                per_gaussian_caches_ptr
            );
        } else {
            CudaRasterizer::Rasterizer::eval_jtf_RHS_only<float>(
                packed_data,
                r_vec.contiguous().data_ptr<float>(),
                sparse_jacobians_ptr,
                index_maps_ptr,
                per_gaussian_caches_ptr
            );
        }
        packed_data.free_pointer_memory(); // Free PackedGSGNDataSpec's internal allocations
    }

    data.free_pointer_memory(); // Free GSGNDataSpec's internal allocations
    cudaFree(index_maps_ptr);
    cudaFree(sparse_jacobians_ptr);
    cudaFree(per_gaussian_caches_ptr);

    return std::make_tuple(r_vec, sparse_jacobians, index_maps, per_gaussian_caches);
}
