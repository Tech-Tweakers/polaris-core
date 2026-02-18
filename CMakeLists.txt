cmake_minimum_required(VERSION 3.16)
project(polaris_core LANGUAGES CXX)

# ============================
# C++ Standard
# ============================
set(CMAKE_CXX_STANDARD 17)
set(CMAKE_CXX_STANDARD_REQUIRED ON)

# ============================
# Paths: llama.cpp root + build selection
# ============================
set(LLAMA_ROOT ${CMAKE_CURRENT_SOURCE_DIR}/../..)

option(POLARIS_ENABLE_CUDA "Enable CUDA backend" OFF)

if(POLARIS_ENABLE_CUDA)
  set(LLAMA_BUILD_DIR ${LLAMA_ROOT}/build-gpu)
else()
  set(LLAMA_BUILD_DIR ${LLAMA_ROOT}/build-cpu)
endif()

set(LLAMA_BIN_DIR ${LLAMA_BUILD_DIR}/bin)

message(STATUS "Using llama build dir: ${LLAMA_BUILD_DIR}")
message(STATUS "Using llama bin dir:   ${LLAMA_BIN_DIR}")

# ============================
# pybind11
# ============================
find_package(pybind11 REQUIRED)

# ============================
# Python module
# ============================
pybind11_add_module(polaris_core
  polaris_bind.cpp
  polaris_gen.cpp
)




# ============================
# Includes (llama headers)
# ============================
target_include_directories(polaris_core PRIVATE
  ${LLAMA_ROOT}
  ${LLAMA_ROOT}/include
  ${LLAMA_ROOT}/common
  ${LLAMA_ROOT}/ggml/include
)

# ============================
# Compiler flags
# ============================
target_compile_options(polaris_core PRIVATE
  -O3 -fPIC -march=native
  -Wall -Wextra -Wno-unused-parameter
)

# ============================
# Threads
# ============================
find_package(Threads REQUIRED)

# ============================
# Standalone linking (IMPORTANT)
# We link against compiled shared libs in build-cpu/bin or build-gpu/bin
# ============================

# Tell linker where libs are
target_link_directories(polaris_core PRIVATE
  ${LLAMA_BIN_DIR}
)

# Link core libs (NO "common", it's not a shared lib)
find_package(CURL REQUIRED)

target_link_libraries(polaris_core PRIVATE
  llama
  ggml-base
  ggml-cpu

  "-Wl,--whole-archive"
  ${LLAMA_BUILD_DIR}/common/libcommon.a
  "-Wl,--no-whole-archive"

  CURL::libcurl
  Threads::Threads
  ${CMAKE_DL_LIBS}
)


# Optional CUDA backend
if(POLARIS_ENABLE_CUDA)
  target_link_libraries(polaris_core PRIVATE ggml-cuda)
  target_compile_definitions(polaris_core PRIVATE POLARIS_USE_CUDA)
endif()

# ============================
# RPATH so Python finds libs at runtime
# ============================
set_target_properties(polaris_core PROPERTIES
  BUILD_RPATH   "${LLAMA_BIN_DIR}"
  INSTALL_RPATH "${LLAMA_BIN_DIR}"
)

message(STATUS "✅ Polaris-Core standalone configured successfully!")
