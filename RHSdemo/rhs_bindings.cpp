#include <pybind11/pybind11.h>
#include <pybind11/stl.h>
#include <pybind11/operators.h>
#include "rhs_functions.h"

namespace py = pybind11;

PYBIND11_MODULE(_C_RHS, m) {
    m.doc() = "Pybind11 bindings for RHS CUDA functions";

    // Bind GSGNDataSpec
    py::class_<GSGNDataSpec>(m, "GSGNDataSpec")
        .def(py::init<>())
        .def_readwrite("background", &GSGNDataSpec::background)
        .def_readwrite("params", &GSGNDataSpec::params)
        .def_readwrite("means3D", &GSGNDataSpec::means3D)
        .def_readwrite("scale_modifier", &GSGNDataSpec::scale_modifier)
        .def_readwrite("viewmatrix", &GSGNDataSpec::viewmatrix)
        .def_readwrite("projmatrix", &GSGNDataSpec::projmatrix)
        .def_readwrite("tan_fovx", &GSGNDataSpec::tan_fovx)
        .def_readwrite("tan_fovy", &GSGNDataSpec::tan_fovy)
        .def_readwrite("cx", &GSGNDataSpec::cx)
        .def_readwrite("cy", &GSGNDataSpec::cy)
        .def_readwrite("sh", &GSGNDataSpec::sh)
        .def_readwrite("unactivated_opacity", &GSGNDataSpec::unactivated_opacity)
        .def_readwrite("unactivated_scales", &GSGNDataSpec::unactivated_scales)
        .def_readwrite("unactivated_rotations", &GSGNDataSpec::unactivated_rotations)
        .def_readwrite("degree", &GSGNDataSpec::degree)
        .def_readwrite("H", &GSGNDataSpec::H)
        .def_readwrite("W", &GSGNDataSpec::W)
        .def_readwrite("campos", &GSGNDataSpec::campos)
        .def_readwrite("geomBuffer", &GSGNDataSpec::geomBuffer)
        .def_readwrite("R", &GSGNDataSpec::R)
        .def_readwrite("binningBuffer", &GSGNDataSpec::binningBuffer)
        .def_readwrite("imageBuffer", &GSGNDataSpec::imageBuffer)
        .def_readwrite("num_sparse_gaussians", &GSGNDataSpec::num_sparse_gaussians)
        .def_readwrite("map_visible_gaussians", &GSGNDataSpec::map_visible_gaussians)
        .def_readwrite("map_cache_to_gaussians", &GSGNDataSpec::map_cache_to_gaussians)
        .def_readwrite("num_visible_gaussians", &GSGNDataSpec::num_visible_gaussians)
        .def_readwrite("debug", &GSGNDataSpec::debug)
        .def_readwrite("use_double_precision", &GSGNDataSpec::use_double_precision)
        .def_readwrite("have_n_contrib_vol_rend_prefix_sum", &GSGNDataSpec::have_n_contrib_vol_rend_prefix_sum)
        .def_readwrite("n_contrib_vol_rend_prefix_sum", &GSGNDataSpec::n_contrib_vol_rend_prefix_sum)
        .def_readwrite("have_residuals", &GSGNDataSpec::have_residuals)
        .def_readwrite("residuals", &GSGNDataSpec::residuals)
        .def_readwrite("have_weights", &GSGNDataSpec::have_weights)
        .def_readwrite("weights", &GSGNDataSpec::weights)
        .def_readwrite("have_residuals_ssim", &GSGNDataSpec::have_residuals_ssim)
        .def_readwrite("residuals_ssim", &GSGNDataSpec::residuals_ssim)
        .def_readwrite("have_weights_ssim", &GSGNDataSpec::have_weights_ssim)
        .def_readwrite("weights_ssim", &GSGNDataSpec::weights_ssim)
        .def_readwrite("P", &GSGNDataSpec::P)
        .def_readwrite("num_images", &GSGNDataSpec::num_images)
        .def_readwrite("num_pixels", &GSGNDataSpec::num_pixels)
        .def_readwrite("jx_stride", &GSGNDataSpec::jx_stride)
        .def_readwrite("M", &GSGNDataSpec::M)
        .def_property_readonly("focal_x", [](const GSGNDataSpec &self) { return self.focal_x; }) // Return as property
        .def_property_readonly("focal_y", [](const GSGNDataSpec &self) { return self.focal_y; }) // Return as property
        .def_readwrite("total_params", &GSGNDataSpec::total_params)
        .def_readwrite("offset_xyz", &GSGNDataSpec::offset_xyz)
        .def_readwrite("offset_scales", &GSGNDataSpec::offset_scales)
        .def_readwrite("offset_rotations", &GSGNDataSpec::offset_rotations)
        .def_readwrite("offset_opacity", &GSGNDataSpec::offset_opacity)
        .def_readwrite("offset_features_dc", &GSGNDataSpec::offset_features_dc)
        .def_readwrite("offset_features_rest", &GSGNDataSpec::offset_features_rest)
        .def("init", &GSGNDataSpec::init)
        .def("free_pointer_memory", &GSGNDataSpec::free_pointer_memory);

    // Bind EvalJTFRHSOnly function
    m.def("eval_jtf_RHS_only", &EvalJTFRHSOnly, "Evaluate JT*F RHS only for SSGN (C++ CUDA)");
}
