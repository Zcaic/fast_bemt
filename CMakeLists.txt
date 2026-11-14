cmake_minimum_required(VERSION 3.26)
project(fast_bemt VERSION 0.1.0 LANGUAGES CXX)

# ==============================================================================
# 1. C++ 标准与基础编译选项
# ==============================================================================
set(CMAKE_CXX_STANDARD 17)
set(CMAKE_CXX_STANDARD_REQUIRED ON)
set(CMAKE_CXX_EXTENSIONS OFF)
set(CMAKE_POSITION_INDEPENDENT_CODE ON)
set(CMAKE_EXPORT_COMPILE_COMMANDS ON) # 生成 compile_commands.json 供 VS Code 代码补全

if(NOT CMAKE_BUILD_TYPE AND NOT CMAKE_CONFIGURATION_TYPES)
    set(CMAKE_BUILD_TYPE Release CACHE STRING "Choose the type of build." FORCE)
endif()

include(FetchContent)

# ==============================================================================
# 2. 离线加载 Eigen3 纯头文件库
# ==============================================================================
FetchContent_Declare(
    Eigen3
    URL "${CMAKE_CURRENT_SOURCE_DIR}/externals/eigen-3.4.0.zip"
    URL_HASH SHA256=EBA3F3D414D2F8CBA2919C78EC6DAAB08FC71BA2BA4AE502B7E5D4D99FC02CDA
    SOURCE_SUBDIR "cmake_dummy_skip" # 故意指定不存在的子目录，仅解压不执行其 CMakeLists.txt
    DOWNLOAD_EXTRACT_TIMESTAMP TRUE
)
FetchContent_MakeAvailable(Eigen3)

# 手动定义干净的 Eigen3::Eigen 纯头文件接口目标
if(NOT TARGET Eigen3::Eigen)
    add_library(Eigen3_Eigen INTERFACE)
    target_include_directories(Eigen3_Eigen SYSTEM INTERFACE "${eigen3_SOURCE_DIR}")
    add_library(Eigen3::Eigen ALIAS Eigen3_Eigen)
endif()

# ==============================================================================
# 3. 编译硬件加速选项 (SIMD: AVX2 + FMA) 与 OpenMP 多线程支持
# ==============================================================================
option(FAST_BEMT_ENABLE_SIMD "Enable AVX2 and FMA vectorization instructions" ON)

add_library(bemt_compiler_flags INTERFACE)

if(MSVC)
    find_package(OpenMP REQUIRED)
    # 显式添加 /utf-8 防止中文注释在 CP936 代码页下触发 C4819 吞换行符错误
    target_compile_options(bemt_compiler_flags INTERFACE /O2 /fp:fast /EHsc /utf-8)
    if(FAST_BEMT_ENABLE_SIMD)
        target_compile_options(bemt_compiler_flags INTERFACE /arch:AVX2)
    endif()
    target_link_libraries(bemt_compiler_flags INTERFACE OpenMP::OpenMP_CXX)

elseif(MINGW)
    # 【关键修改】MinGW 下不走 find_package(OpenMP) 的链接目标，防止引入 libgomp.dll.a
    # 附加 -fno-finite-math-only 确保 -ffast-math 下 std::isfinite() 与 allFinite() 保护不被编译器优化失效
    target_compile_options(bemt_compiler_flags INTERFACE -O3 -ffast-math -fno-finite-math-only -Wall -Wextra -fopenmp)
    if(FAST_BEMT_ENABLE_SIMD)
        target_compile_options(bemt_compiler_flags INTERFACE -march=native -mavx2 -mfma)
    endif()
    # 静态链接标准库运行时
    target_link_options(bemt_compiler_flags INTERFACE -static-libgcc -static-libstdc++)
    # 放在 target_link_libraries 中，确保在 bemt_core 静态库【之后】被链接器扫描
    target_link_libraries(bemt_compiler_flags INTERFACE
        "-Wl,-Bstatic" gomp winpthread "-Wl,-Bdynamic"
    )

else()
    # Linux / macOS
    find_package(OpenMP REQUIRED)
    target_compile_options(bemt_compiler_flags INTERFACE -O3 -ffast-math -fno-finite-math-only -Wall -Wextra)
    if(FAST_BEMT_ENABLE_SIMD)
        target_compile_options(bemt_compiler_flags INTERFACE -march=native -mavx2 -mfma)
    endif()
    target_link_libraries(bemt_compiler_flags INTERFACE OpenMP::OpenMP_CXX)
endif()

# ==============================================================================
# 4. 构建 C++ 核心静态库 (bemt_core)
# ==============================================================================
set(BEMT_CORE_SOURCES
    src/airfoil.cpp
    src/solver.cpp
    src/sensitivity.cpp
)

set(BEMT_BINDING_SOURCES
    src/bindings.cpp
)

add_library(bemt_core STATIC ${BEMT_CORE_SOURCES})
target_include_directories(bemt_core PUBLIC
    $<BUILD_INTERFACE:${CMAKE_CURRENT_SOURCE_DIR}/include>
)
# 所有 OpenMP 与硬件加速配置已集成在 bemt_compiler_flags 中
target_link_libraries(bemt_core PUBLIC
    Eigen3::Eigen
    bemt_compiler_flags
)

# ==============================================================================
# 5. 构建 nanobind Python 扩展模块 (_fast_bemt_impl)
# ==============================================================================
find_package(Python 3.9 COMPONENTS Interpreter Development.Module OPTIONAL_COMPONENTS Development.SABIModule REQUIRED)

FetchContent_Declare(
    nanobind
    URL "${CMAKE_CURRENT_SOURCE_DIR}/externals/nanobind.zip"
    URL_HASH SHA256=AF87F5EBB5386CA6D09B9A525AF18FC080BBF9C61BF5B0480B3EB17A9C4B8A8E
    DOWNLOAD_EXTRACT_TIMESTAMP TRUE
)
FetchContent_MakeAvailable(nanobind)

nanobind_add_module(
    _fast_bemt_impl
    STABLE_ABI
    NB_STATIC
    ${BEMT_BINDING_SOURCES}
)

# 链接 bemt_core 时，会自动按正确顺序继承 bemt_compiler_flags 中的静态 gomp 设置
target_link_libraries(_fast_bemt_impl PRIVATE bemt_core)

# 1. 编译时直接将 .pyd/.so 输出到 python/fast_bemt 目录下（使用 $<1:...> 防止多配置生成器追加 Release 子目录）
set_target_properties(_fast_bemt_impl PROPERTIES
    LIBRARY_OUTPUT_DIRECTORY "$<1:${CMAKE_CURRENT_SOURCE_DIR}/python/fast_bemt>"
    RUNTIME_OUTPUT_DIRECTORY "$<1:${CMAKE_CURRENT_SOURCE_DIR}/python/fast_bemt>"
)

# 2. 若执行 cmake --install build，同样安装到 python/fast_bemt 目录下
install(TARGETS _fast_bemt_impl
    LIBRARY DESTINATION "${CMAKE_CURRENT_SOURCE_DIR}/python/fast_bemt"
    RUNTIME DESTINATION "${CMAKE_CURRENT_SOURCE_DIR}/python/fast_bemt"
)