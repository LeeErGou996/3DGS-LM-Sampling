import os
from setuptools import setup
from torch.utils.cpp_extension import CUDAExtension, BuildExtension

# Get the directory of this setup.py file
current_dir = os.path.dirname(os.path.abspath(__file__))

setup(
    name="rhs_cuda_extension",
    version="0.1.0",
    ext_modules=[
        CUDAExtension(
            name="rhs_cuda_extension._C_RHS",
            sources=[
                os.path.join(current_dir, "rhs_functions.cu"),
                os.path.join(current_dir, "rhs_bindings.cpp")
            ],
            extra_compile_args={
                "cxx": ["-g"],
                "nvcc": [
                    "-I" + os.path.join(os.path.dirname(os.path.dirname(os.path.abspath(__file__))), "submodules/diff-gaussian-rasterization/third_party/glm/"),
                    "-I" + os.path.join(os.path.dirname(os.path.dirname(os.path.abspath(__file__))), "submodules/diff-gaussian-rasterization/cuda_rasterizer/"),
                    "--use_fast_math",
                    "-lineinfo",
                    "--generate-line-info"
                ]
            },
        )
    ],
    cmdclass={
        'build_ext': BuildExtension
    },
    install_requires=[
        "torch>=1.8.0",
        "numpy"
    ],
)