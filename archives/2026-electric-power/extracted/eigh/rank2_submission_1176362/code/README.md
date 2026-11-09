# 算子编译

安装 CANN Toolkit 9.0 或兼容版本及 CMake，然后执行：

```bash
source /path/to/cann/set_env.sh
cmake -S . -B build -DCANNJUDGE_COMPILE_JOBS=4
cmake --build build --parallel 4 --target binary package
```

编辑 op_host 和 op_kernel 下的源码。安装包生成在 build 目录中。
工程使用生成时指定的芯片型号；本工程提供编译流程。
