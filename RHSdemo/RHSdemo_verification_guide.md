# SSGN RHS 验证 Demo 配置与运行步骤

为了验证您修改后的 C++/CUDA 代码与 Python 逻辑（特别是 SSGN 策略）是否能顺利联通且无内存越界 Bug，请按照以下步骤操作：

### 1. 确保文件就位

请确保以下文件已放置在 `C:\InDeutschland\3IN2390\kaggle\3DGS-LM\RHSdemo\` 目录下：

*   `cpp_cuda_functions.cpp`
*   `python_functions.py`
*   `setup.py`
*   `verify_ssgn.py`

### 2. 编译 C++/CUDA 扩展

在命令行中，导航到 `RHSdemo` 目录，并执行 `setup.py` 来编译 C++/CUDA 扩展。

```bash
cd C:\InDeutschland\3IN2390\kaggle\3DGS-LM\RHSdemo
python setup.py install
```

**说明:**
*   `python setup.py install` 命令会编译 `cpp_cuda_functions.cpp` 并将其安装为一个 Python 模块 `rhs_cuda_extension._C_RHS`。
*   这个过程可能需要几分钟，具体取决于您的系统性能和 CUDA 环境配置。
*   编译过程中可能会出现警告，但只要没有致命错误导致编译失败即可。

### 3. 运行验证脚本

编译和安装完成后，您可以通过运行 `verify_ssgn.py` 脚本来执行 demo：

```bash
python verify_ssgn.py
```

**预期结果:**

如果一切顺利，脚本将输出：
*   关于 demo 运行的进度信息。
*   一条成功消息，表明 `linear_solve_pcg_fused` 函数已成功执行。
*   `result` 字典中的键和 `x` (更新向量) 的形状。
*   如果 `timing_dict` 有数据，还会显示时序信息。

如果出现错误，脚本会捕获异常并打印失败消息。这可能表明 C++/CUDA 代码、绑定或模拟数据存在问题。

**排查提示:**

*   **CUDA 不可用**: 如果您看到 "CUDA is not available" 的提示，请确保您的 PyTorch 安装支持 CUDA，并且您的 GPU 驱动程序已正确安装。
*   **编译错误**: 如果 `python setup.py install` 失败，请仔细检查错误日志，可能是因为缺少 CUDA Toolkit、PyTorch Headers 或其他依赖项。
*   **运行时错误**: 如果 `verify_ssgn.py` 运行时崩溃，请查看详细的 Python 栈回溯和任何 C++/CUDA 错误信息（如果 `pipe.debug=True` 启用）。

请按照上述步骤进行操作。如果您在任何步骤中遇到问题，请告诉我，我将尽力协助您排查。
