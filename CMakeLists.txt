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
# 3. 查找 OpenMP 多线程支持
# ==============================================================================
find_package(OpenMP REQUIRED)

# ==============================================================================
# 4. 编译硬件加速选项 (SIMD: AVX2 + FMA)
# ==============================================================================
option(FAST_BEMT_ENABLE_SIMD "Enable AVX2 and FMA vectorization instructions" ON)

add_library(bemt_compiler_flags INTERFACE)
if(MSVC)
    target_compile_options(bemt_compiler_flags INTERFACE /O2 /fp:fast /EHsc)
    if(FAST_BEMT_ENABLE_SIMD)
        target_compile_options(bemt_compiler_flags INTERFACE /arch:AVX2)
    endif()
else()
    # GCC / Clang / MinGW-w64 (UCRT64)
    target_compile_options(bemt_compiler_flags INTERFACE -O3 -ffast-math -Wall -Wextra)
    if(FAST_BEMT_ENABLE_SIMD)
        target_compile_options(bemt_compiler_flags INTERFACE -march=native -mavx2 -mfma)
    endif()
endif()

# ==============================================================================
# 5. 构建 C++ 核心静态库 (bemt_core)
# ==============================================================================
set(BEMT_CORE_SOURCES
    src/airfoil.cpp
    src/solver.cpp
    src/sensitivity.cpp
)

set(BEMT_BINDING_SOURCES
    src/bindings.cpp
)

# 自动生成尚未编写的源文件占位符，确保分步开发时 CMake 随时可以 configure 通过
foreach(src_file IN LISTS BEMT_CORE_SOURCES BEMT_BINDING_SOURCES)
    if(NOT EXISTS "${CMAKE_CURRENT_SOURCE_DIR}/${src_file}")
        get_filename_component(src_dir "${CMAKE_CURRENT_SOURCE_DIR}/${src_file}" DIRECTORY)
        file(MAKE_DIRECTORY "${src_dir}")
        file(WRITE "${CMAKE_CURRENT_SOURCE_DIR}/${src_file}" "// Placeholder for ${src_file}\n")
    endif()
endforeach()

add_library(bemt_core STATIC ${BEMT_CORE_SOURCES})
target_include_directories(bemt_core PUBLIC
    $<BUILD_INTERFACE:${CMAKE_CURRENT_SOURCE_DIR}/include>
)
target_link_libraries(bemt_core PUBLIC
    Eigen3::Eigen
    OpenMP::OpenMP_CXX
    bemt_compiler_flags
)

# ==============================================================================
# 6. 构建 nanobind Python 扩展模块 (_fast_bemt_impl)
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

target_link_libraries(_fast_bemt_impl PRIVATE bemt_core)

# 在 MinGW-w64 下静态链接标准库与 OpenMP/pthread 运行时，彻底消除 Python 导入 .pyd 时的 DLL 缺失问题
if(MINGW)
    target_link_options(_fast_bemt_impl PRIVATE
        -static-libgcc
        -static-libstdc++
        -Wl,-Bstatic -lgomp -lwinpthread -Wl,-Bdynamic
    )
endif()

# 安装扩展模块到 fast_bemt Python 包目录下
install(TARGETS _fast_bemt_impl LIBRARY DESTINATION fast_bemt)