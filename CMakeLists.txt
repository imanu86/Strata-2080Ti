# Strata Engine - top-level build.
#
# One project builds the engine for every backend: CUDA (STRATA_ENABLE_CUDA), HIP (STRATA_ENABLE_HIP; the layer-split
# and DeepSeek engines are the STRATA_BUILD_SPLIT / STRATA_BUILD_DSV41 options), the CPU-only build, and the test
# programs.  The Intel Arc (SYCL) engine is a separate project in sycl/ (STRATA_ENABLE_SYCL), and the Metal engine
# one in metal/.  CMakePresets.json has the CPU check preset (`cmake --preset cpu-check`) and the compile-only
# backend presets used by tools/check and the CI workflows; the release scripts in packaging/ pass their own flags.
#
# The CUDA kernels need sm_80 or newer (tf32 mma, bf16 math) and the guard below refuses anything older: silently
# building for a pre-Ampere arch would produce a binary that runs and is wrong about its own performance, which is
# the worst of both outcomes.
cmake_minimum_required(VERSION 3.24)
project(strata VERSION 0.1.59.1 LANGUAGES CXX)
include(${CMAKE_CURRENT_SOURCE_DIR}/cmake/strata_geometry.cmake)   # -DSTRATA_GEOMETRY=<name>: the model shape the kernels are compiled for (default flash_next, no flag added)
# The build is cut by backend (cleanup 5.4): each file below only DEFINES macros (strata_<part>()), which run here, in this scope, at the place their
# block used to be - so this file reads in the order the old single CMakeLists.txt evaluated.
include(${CMAKE_CURRENT_SOURCE_DIR}/cmake/strata_ggml.cmake)          # which ggml: the vendored third_party/strata-ggml, or -DSTRATA_GGML_DIR
include(${CMAKE_CURRENT_SOURCE_DIR}/cmake/backends/cpu.cmake)         # strata_kernels_cpu, the native (ggml-cpu) expert rows, the CPU tests
include(${CMAKE_CURRENT_SOURCE_DIR}/cmake/backends/cuda.cmake)        # NVIDIA: toolchain, llama.cpp's MMQ kernels (strata_mmq, strata_kq)
include(${CMAKE_CURRENT_SOURCE_DIR}/cmake/backends/hip.cmake)         # AMD: gfx906 compat, MMQ on HIP
include(${CMAKE_CURRENT_SOURCE_DIR}/cmake/strata_gpu_engine.cmake)    # the GPU engine both of them build
# EXPERIMENTAL: the Intel Arc engine, the SYCL port in sycl/ (PR #423; icpx + oneMKL, docs/INTEL_ARC.md). It is its
# own project that takes the unported sources from src/; with this on, nothing below is configured.
option(STRATA_ENABLE_SYCL "EXPERIMENTAL: build the Intel Arc (SYCL) engine from sycl/ - docs/INTEL_ARC.md" OFF)
if(STRATA_ENABLE_SYCL)
  if(STRATA_ENABLE_CUDA OR STRATA_ENABLE_HIP)
    message(FATAL_ERROR "STRATA_ENABLE_SYCL is a separate backend; leave STRATA_ENABLE_CUDA and STRATA_ENABLE_HIP off")
  endif()
  message(WARNING "STRATA_ENABLE_SYCL: the Intel Arc engine is experimental (community port, docs/INTEL_ARC.md)")
  add_subdirectory(sycl)
  return()
endif()
add_compile_definitions(STRATA_VERSION="${PROJECT_VERSION}")
# issue #31: release builds keep a PDB beside the exe (not shipped in it), so a minidump from a stalled engine can
# be read back to the exact source line
if(MSVC AND CMAKE_BUILD_TYPE STREQUAL "Release")
  add_compile_options($<$<COMPILE_LANGUAGE:CXX>:/Zi>)
  add_link_options(/DEBUG /OPT:REF /OPT:ICF)
endif()

# The prebuilt release (setup.py downloads it, so users need no compiler): ggml's CPU backend for any AVX2 CPU
# instead of this machine's (Strata's own AVX-512 kernels are per-file and chosen at run time either way), and
# the MSVC runtime linked in, so no Visual C++ redistributable is needed.
option(STRATA_PORTABLE "build for other PCs: AVX2 baseline, static MSVC runtime" OFF)
# EXPERIMENTAL, opt-in: an engine for CPUs without AVX2 (#394, #595, #623), compiled on that PC.  ggml-cpu, which then
# computes every CPU expert row, is built for this floor instead of AVX2 or the build host: "avx" (Sandy/Ivy Bridge:
# AVX, no FMA/F16C/AVX2) or "none" (SSE4.2 + POPCNT: Nehalem/Westmere).  Strata's own AVX2/AVX-512 kernels stay in the
# binary behind their CPU checks; the router lookahead gets an AVX1 kernel.  Empty (the default) changes nothing.
set(STRATA_ISA_FLOOR "" CACHE STRING "experimental older-CPU build: empty (default, AVX2 floor), avx, or none (SSE4.2)")
set_property(CACHE STRATA_ISA_FLOOR PROPERTY STRINGS "" avx none)
if(NOT STRATA_ISA_FLOOR STREQUAL "" AND NOT STRATA_ISA_FLOOR STREQUAL "avx" AND NOT STRATA_ISA_FLOOR STREQUAL "none")
  message(FATAL_ERROR "STRATA_ISA_FLOOR must be empty, avx or none (got '${STRATA_ISA_FLOOR}')")
endif()
if(STRATA_ISA_FLOOR AND STRATA_PORTABLE)
  message(FATAL_ERROR "STRATA_ISA_FLOOR (an older-CPU build) and STRATA_PORTABLE (the AVX2 release build) exclude each other")
endif()
if(STRATA_PORTABLE AND MSVC)
  set(CMAKE_MSVC_RUNTIME_LIBRARY "MultiThreaded")
endif()

# #3202: Windows hands a program its command line in the system's ANSI code page (cp1251 / cp866 on a Russian Windows, cp1252 on an
# English one), but the engine reads every path as UTF-8 (the run configs and the server write UTF-8). A model folder with
# Cyrillic letters arrived as other bytes, or as "?" where the page has no such letter, and the model did not open. This manifest
# makes the process's ANSI code page UTF-8 (Windows 10 1903 and newer; older Windows ignores it and nothing changes), so the
# command line, fopen and std::ifstream all take UTF-8.
function(strata_utf8_manifest target)
  if(MSVC)
    target_sources(${target} PRIVATE ${CMAKE_CURRENT_FUNCTION_LIST_DIR}/cmake/strata-utf8.manifest)
  endif()
endfunction()

# The AVX kernels exist for x86 only.  On aarch64 (DGX Spark / GB10) portable.cpp stands in for them and
# native experts run on ggml-cpu's NEON dot products.
if(NOT CMAKE_SYSTEM_PROCESSOR OR CMAKE_SYSTEM_PROCESSOR MATCHES "^(x86_64|AMD64|amd64|x64|X64|i[3-6]86|x86|X86)$")   # an unset processor stays the x86 build, as before
  set(STRATA_CPU_X86 ON)
else()
  set(STRATA_CPU_X86 OFF)
endif()
if(STRATA_ISA_FLOOR AND NOT STRATA_CPU_X86)
  message(WARNING "STRATA_ISA_FLOOR is an x86 older-CPU build and does not apply to ${CMAKE_SYSTEM_PROCESSOR}; ignored")
  set(STRATA_ISA_FLOOR "" CACHE STRING "" FORCE)
endif()
set(CMAKE_CXX_STANDARD 20)
set(CMAKE_CXX_STANDARD_REQUIRED ON)
set(CMAKE_CXX_EXTENSIONS OFF)
if(NOT CMAKE_BUILD_TYPE AND NOT CMAKE_CONFIGURATION_TYPES)
  set(CMAKE_BUILD_TYPE Release CACHE STRING "" FORCE)
endif()

option(STRATA_ENABLE_CUDA "Build the CUDA targets (Phase 2+)" OFF)
# AMD gfx906 (MI50, wave64): the same .cu sources compiled as HIP through the gfx906 compat layer.  The CUDA runtime and
# cuBLAS names are mapped by the headers in include/strata/platform/hip_compat/, which go first on the include path,
# and CUDA::cudart / CUDA::cublas become aliases of HIP's targets, so every CUDA target below stays as it is.  This is
# separate from STRATA_ENABLE_HIP (the wave32 RDNA backend): with it on, STRATA_ENABLE_CUDA is on and
# STRATA_ENABLE_HIP stays off, so every "CUDA" branch of the tree is the one taken.
option(STRATA_HIP_GFX906 "Build the CUDA targets for AMD gfx906 through HIP (wave64 compat layer)" OFF)
if(STRATA_HIP_GFX906)
  if(STRATA_ENABLE_HIP)
    message(FATAL_ERROR "STRATA_HIP_GFX906 and STRATA_ENABLE_HIP are separate backends; enable only one")
  endif()
  set(STRATA_ENABLE_CUDA ON CACHE BOOL "" FORCE)
endif()
option(STRATA_ENABLE_HIP "Build the HIP targets for wave32 AMD GPUs (gfx1100, gfx1201; gfx1012/gfx1030 unvalidated)" OFF)
option(STRATA_PREFILL_MMQ "Build the optional GGML MMQ prefill path for HIP" OFF)
# The community-made AMD changes (include/strata/core/amd_community.hpp): on by default.  This option is the build half of the one
# switch STRATA_AMD_COMMUNITY_TUNING: OFF compiles the portable signed-byte dot for gfx1010 again (the SDWA sequence of #4855).
# The rocBLAS tables are switched at run time (STRATA_AMD_COMMUNITY_TUNING=0).
option(STRATA_AMD_COMMUNITY_TUNING "Compile the community-made gfx1010 SDWA dot product (the runtime switch of the same name controls the gfx1030 rocBLAS tables)" ON)
if(NOT STRATA_AMD_COMMUNITY_TUNING)
  add_compile_definitions(STRATA_AMD_COMMUNITY_DOT_OFF=1)
endif()
# The rows one verify / batch window holds (include/strata/kernels/window_limits.hpp): 8, the 0.2.7 window, or 16 (four pipelined
# groups of 16 slots = 64 slots on a four-card split, docs/BATCHING.md).  Only the CUDA and HIP engines are built from
# these sources with a wide window; the SYCL and Metal builds never read this variable and keep 8.  A window of up to
# 8 rows runs the kernels it always ran, whichever is chosen (the wide instantiations are extra code).  "auto" = 16 for
# the HIP engines (a HIP code object loads on first use, so the extra kernels cost nothing until a wide window forms)
# and 8 for CUDA, which loads every kernel at start (CUDA_MODULE_LOADING=EAGER: the wide kernels would take VRAM from
# the expert cache of every card); -DSTRATA_WINDOW_MAX_T=16 builds the wide CUDA engine.
set(STRATA_WINDOW_MAX_T "auto" CACHE STRING "rows of a verify / batch window: 8, 12, 16 or auto (HIP 16, CUDA 8)")
set_property(CACHE STRATA_WINDOW_MAX_T PROPERTY STRINGS auto 8 12 16)
if(NOT STRATA_WINDOW_MAX_T STREQUAL "auto" AND NOT STRATA_WINDOW_MAX_T STREQUAL "8" AND NOT STRATA_WINDOW_MAX_T STREQUAL "12" AND NOT STRATA_WINDOW_MAX_T STREQUAL "16")
  message(FATAL_ERROR "STRATA_WINDOW_MAX_T must be auto, 8, 12 or 16 (got '${STRATA_WINDOW_MAX_T}')")
endif()
if(STRATA_ENABLE_HIP OR STRATA_ENABLE_CUDA)
  if(STRATA_WINDOW_MAX_T STREQUAL "auto")
    if(STRATA_ENABLE_HIP)
      set(_strata_window_max_t 16)
    else()
      set(_strata_window_max_t 8)
    endif()
  else()
    set(_strata_window_max_t ${STRATA_WINDOW_MAX_T})
  endif()
  add_compile_definitions(STRATA_VERIFY_MAX_T=${_strata_window_max_t})
  message(STATUS "Strata: verify / batch windows hold up to ${_strata_window_max_t} rows")
endif()
# MMQ prompt kernels for Q4_K / Q5_K / Q5_1 / Q6_K (Unsloth's UD-Q4/Q5/Q6_K_XL experts; on HIP also the dense GGUF
# projections, STRATA_DENSE_MMQ).  On by default since MODEL-UPDATE Phase 1: on CUDA they are their own library
# (strata_kq, loaded only for a model with these formats: the engine loads every kernel of its own binary at start,
# CUDA_MODULE_LOADING=EAGER, and these took ~3.8 MB of VRAM - 3 IQ3_XXS expert slots on a P100 - from every model); on
# HIP a code object loads on first use, so they cost other models nothing there (R9700: the same slots).  Used at run
# time when STRATA_MMQ_KQ=1, which the engine sets by itself on NVIDIA cards of compute capability 8.0 and up (include/strata/core/nv_defaults.hpp) and on
# Linux gfx12 (STRATA_MMQ_KQ=0 turns it off; setup also writes it for UD-Q5_K_XL / UD-Q6_K_XL); other cards keep the FP16 path.  This CMake option only
# decides whether the kernels are BUILT: on in every release build.
option(STRATA_MMQ_TQ_OVERLAY "Gyro trellis MMQ prompt kernels: the patched ggml-cuda headers of third_party/strata-ggml/ggml/src/ggml-cuda-tq (with -DSTRATA_GGML_DIR: an overlay of that llama.cpp's headers written at configure time, cmake/strata_ggml_tq_overlay.cmake)" ON)
option(STRATA_MMQ_KQUANTS "Build the MMQ prompt kernels for Q4_K / Q5_K / Q5_1 / Q6_K: the experts of Unsloth's K-quant files, and on HIP the dense GGUF projections (STRATA_DENSE_MMQ)" ON)
# Q6_K gate/up in the grouped expert kernels (Unsloth's UD-Q6_K_XL, and layer 2 of UD-Q5_K_XL; docs/QUANTS.md).  On by
# default since MODEL-UPDATE Phase 1: one more instance per grouped kernel in iq_kernels.cu, +19 KB of sm_60 code,
# measured to leave every expert-cache slot count as it was (P100, R9700).  -DSTRATA_Q6K_EXPERTS=OFF drops it.
option(STRATA_Q6K_EXPERTS "Build the Q6_K gate/up grouped expert kernels (UD-Q6_K_XL, UD-Q5_K_XL)" ON)
if(STRATA_Q6K_EXPERTS)
  add_compile_definitions(STRATA_Q6K_EXPERTS=1)
endif()
option(STRATA_ORCA_Q4KS_MMQ "Build the GGML Q5_0 MMQ instance for OrcaRouter Q4_K_S" OFF)
# CUDA: the pointer-table MMQ behind STRATA_MMQ_DIRECT=1 (the prompt path's MMQ reads resident experts in their cache
# slots through a Strata-side copy of llama.cpp's MMQ scheduler, src/prefill/moe_mmq_tab.cuh).  Off by default, like
# the K-quant options above: the engine loads every kernel when it starts (CUDA_MODULE_LOADING=EAGER), and these nine
# instances (+15 MB of binary) take 36-38 MiB of VRAM per GPU on sm_86 - expert slots every model would lose, with
# STRATA_MMQ_DIRECT unset too.  They also add ~70% of the upstream MMQ instances' compile time per CUDA arch (175 s
# of CPU for the nine at sm_86 alone).  Off, the build's device code is byte for byte the one without this feature.  CUDA
# only (not HIP, not gfx906); switched off at configure time when the llama.cpp tree's mmq.cuh is not one the copy
# was verified against.
option(STRATA_MMQ_TAB "Build the pointer-table prompt MMQ for STRATA_MMQ_DIRECT=1 (CUDA; ~37 MiB VRAM per GPU)" OFF)
if(STRATA_PREFILL_MMQ AND NOT STRATA_ENABLE_HIP)
  message(FATAL_ERROR "STRATA_PREFILL_MMQ requires STRATA_ENABLE_HIP=ON")
endif()
if(STRATA_ENABLE_CUDA AND STRATA_ENABLE_HIP AND NOT STRATA_HIP_GFX906)
  message(FATAL_ERROR "STRATA_ENABLE_CUDA and STRATA_ENABLE_HIP are separate backends; enable only one")
endif()
if(STRATA_ENABLE_HIP)
  include(cmake/hip_backend.cmake)
endif()
# Tests are on by default when the build can run them.  What they need: their sources (src/**/*_test.cpp, tests/core,
# bench/micro; a source drop without them has nothing to build) and the x86 CPU kernels, which most of the CPU suite drives
# (arm64 stays off by default: -DSTRATA_BUILD_TESTS=ON still builds what builds there).  GPU tests are registered only
# when STRATA_ENABLE_CUDA / STRATA_ENABLE_HIP is on, and the configure summary below says which kinds were registered, so
# a missing GPU is a stated skip and never a silent pass.  Release builds say -DSTRATA_BUILD_TESTS=OFF themselves
# (packaging/, setup.py, Dockerfile, spark/).
set(_strata_tests_default OFF)
if(EXISTS ${CMAKE_CURRENT_SOURCE_DIR}/tests/core AND EXISTS ${CMAKE_CURRENT_SOURCE_DIR}/bench/micro AND STRATA_CPU_X86)
  set(_strata_tests_default ON)
endif()
option(STRATA_BUILD_TESTS "Build the test suite" ${_strata_tests_default})
# A test that needs a model file (a real expert blob from a pack, which is not in the source tree): registered as the real test
# when the file is there, and otherwise as a stated skip that ctest lists as "Skipped" with this reason, never as a pass of nothing
# and not as a failure on a machine that has no pack.
function(strata_add_pack_test name needs)
  if(EXISTS "${CMAKE_CURRENT_SOURCE_DIR}/${needs}")
    add_test(NAME ${name} COMMAND ${ARGN} WORKING_DIRECTORY ${CMAKE_CURRENT_SOURCE_DIR})
  else()
    add_test(NAME ${name} COMMAND ${CMAKE_COMMAND} -E echo "SKIPPED: ${name} needs ${needs} (a Q2_0 expert pack, a model file), which is not in this tree")
    set_tests_properties(${name} PROPERTIES SKIP_REGULAR_EXPRESSION "SKIPPED: ")
  endif()
endfunction()
option(STRATA_BUILD_CONVERSATION_TESTS "Build standalone conversation cache tests" OFF)
option(STRATA_WERROR "Treat compiler warnings as errors" OFF)
# #2027: defined here, before the GPU section reads it to register the native-expert tests (it was defined at the
# native-experts section far below, so a first configure registered four tests fewer than the second)
option(STRATA_NATIVE_EXPERTS "Native (IQ) expert support through ggml-cpu" ON)

# fork: the system prefix persisted on disk (--prompt-cache-file)
add_library(strata_prefix_cache STATIC src/core/prefix_cache_file.cpp)
target_include_directories(strata_prefix_cache PUBLIC include)
if(WIN32)
  target_link_libraries(strata_prefix_cache PRIVATE bcrypt)
endif()

if(STRATA_BUILD_CONVERSATION_TESTS)
  enable_testing()
  add_executable(prefix_cache_file_test src/core/prefix_cache_file_test.cpp)
  target_link_libraries(prefix_cache_file_test PRIVATE strata_prefix_cache)
  add_test(NAME prefix_cache_file_test COMMAND prefix_cache_file_test)
  add_executable(prefix_cache_portable_test src/core/prefix_cache_file_test.cpp src/core/prefix_cache_file.cpp)
  target_include_directories(prefix_cache_portable_test PRIVATE include)
  target_compile_definitions(prefix_cache_portable_test PRIVATE STRATA_PREFIX_PORTABLE_SHA=1)
  add_test(NAME prefix_cache_portable_test COMMAND prefix_cache_portable_test)
  set_tests_properties(prefix_cache_file_test prefix_cache_portable_test PROPERTIES TIMEOUT 60)
  add_executable(conversation_cache_test src/core/conversation_cache_test.cpp)
  target_include_directories(conversation_cache_test PRIVATE include)
  add_test(NAME conversation_cache_test COMMAND conversation_cache_test)
  add_executable(conversation_split_failure_test src/core/conversation_split_failure_test.cpp)
  target_include_directories(conversation_split_failure_test PRIVATE include)
  add_test(NAME conversation_split_failure_test COMMAND conversation_split_failure_test)
  add_executable(conversation_memory_test src/core/conversation_memory_test.cpp src/core/conversation_memory.cpp)
  target_include_directories(conversation_memory_test PRIVATE include)
  add_test(NAME conversation_memory_test COMMAND conversation_memory_test)
  add_executable(conversation_file_test src/core/conversation_file_test.cpp src/core/conversation_file.cpp)
  target_include_directories(conversation_file_test PRIVATE include)
  add_test(NAME conversation_file_test COMMAND conversation_file_test)
  add_executable(session_save_reclaim_test src/core/session_save_reclaim_test.cpp src/core/conversation_file.cpp)
  target_include_directories(session_save_reclaim_test PRIVATE include)
  add_test(NAME session_save_reclaim_test COMMAND session_save_reclaim_test)
endif()

if(STRATA_HIP_GFX906)
  strata_hip_gfx906_toolchain()
elseif(STRATA_ENABLE_CUDA)
  strata_cuda_toolchain()
elseif(STRATA_ENABLE_HIP)
  set(_strata_gpu_runtime_target strata_hip_runtime)
  set(_strata_gpu_blas_target ${STRATA_HIP_BLAS_TARGETS})
  set(_strata_gpu_include_directories ${STRATA_HIP_COMPAT_INCLUDE_DIR})
endif()

# Warnings live on an INTERFACE target the project's own code links, NOT in add_compile_options: applied
# globally they would also hit FetchContent'd Catch2, whose -W4 warnings are not this project's to fix and
# whose /WX failures would be indistinguishable from our own.
#
# Every flag is guarded by COMPILE_LANGUAGE:CXX.  Without that guard nvcc receives bare `/W4 /utf-8` and dies
# with "A single input file is required for a non-link phase" - it parses them as inputs rather than host
# flags.  Host flags for CUDA must go through -Xcompiler, which is what COMPILE_LANGUAGE:CUDA does below.
add_library(strata_warnings INTERFACE)
target_compile_options(strata_warnings INTERFACE
  $<$<AND:$<COMPILE_LANGUAGE:CXX>,$<CXX_COMPILER_ID:MSVC>>:/W4>
  $<$<AND:$<COMPILE_LANGUAGE:CXX>,$<CXX_COMPILER_ID:MSVC>>:/utf-8>
  $<$<AND:$<COMPILE_LANGUAGE:CXX>,$<NOT:$<CXX_COMPILER_ID:MSVC>>>:-Wall>
  $<$<AND:$<COMPILE_LANGUAGE:CXX>,$<NOT:$<CXX_COMPILER_ID:MSVC>>>:-Wextra>
  $<$<AND:$<COMPILE_LANGUAGE:CUDA>,$<CXX_COMPILER_ID:MSVC>>:-Xcompiler=/W4>
  $<$<COMPILE_LANGUAGE:HIP>:-Wall>
  $<$<COMPILE_LANGUAGE:HIP>:-Wextra>)
if(STRATA_WERROR)
  target_compile_options(strata_warnings INTERFACE
    $<$<AND:$<COMPILE_LANGUAGE:CXX>,$<CXX_COMPILER_ID:MSVC>>:/WX>
    $<$<AND:$<COMPILE_LANGUAGE:CXX>,$<NOT:$<CXX_COMPILER_ID:MSVC>>>:-Werror>
    $<$<COMPILE_LANGUAGE:HIP>:-Werror>)
endif()

# ---- strata_artifact -------------------------------------------------------
# Header-only, and that is a deliberate property rather than an unfinished split: the GGUF reader and the
# scalar dequantizers hold no state worth hiding, and a header-only boundary cannot introduce a
# duplicate-symbol or missing-declaration bug into code that is already validated against the real artifact.
add_library(strata_artifact INTERFACE)
target_include_directories(strata_artifact INTERFACE ${CMAKE_CURRENT_SOURCE_DIR}/include)
target_link_libraries(strata_artifact INTERFACE strata_warnings)

add_executable(strata-gguf src/artifact/gguf_reader.cpp)
target_link_libraries(strata-gguf PRIVATE strata_artifact)

add_executable(strata-dequant src/artifact/dequant.cpp)
target_link_libraries(strata-dequant PRIVATE strata_artifact)
# this file is now nothing but the CLI entry point, and its inherited guard spells that as a "selftest"
target_compile_definitions(strata-dequant PRIVATE STRATA_DEQUANT_SELFTEST)
if(STRATA_BUILD_TESTS)
  add_executable(batch_telemetry_test src/core/batch_telemetry_test.cpp)
  target_include_directories(batch_telemetry_test PRIVATE include)
  add_test(NAME batch_telemetry_test COMMAND batch_telemetry_test)
  add_executable(logit_bias_test src/core/logit_bias_test.cpp)
  target_include_directories(logit_bias_test PRIVATE include)
  add_test(NAME logit_bias_test COMMAND logit_bias_test)
  # the reader refuses a duplicate tensor name at open, naming it (a synthetic GGUF: no model, no GPU)
  add_executable(gguf_reader_test src/artifact/gguf_reader_test.cpp)
  target_link_libraries(gguf_reader_test PRIVATE strata_artifact)
  add_test(NAME gguf_reader_test COMMAND gguf_reader_test)
  # a split model's shards and their joint view: missing shard, split keys, duplicates, truncation (synthetic)
  add_executable(gguf_split_test tests/core/gguf_split_test.cpp)
  target_link_libraries(gguf_split_test PRIVATE strata_artifact)
  add_test(NAME gguf_split_test COMMAND gguf_split_test)
  # the model families (qwen4exp, qwen35moe): the guards, the geometry from metadata, the layout tables (synthetic)
  add_executable(model_family_test tests/core/model_family_test.cpp)
  target_link_libraries(model_family_test PRIVATE strata_artifact)
  add_test(NAME model_family_test COMMAND model_family_test)
  # how --batch cuts a prompt read into pieces (a layer split with no slot decoding must still reach a BYIELD)
  add_executable(batch_read_test tests/program/batch_read_test.cpp)
  target_include_directories(batch_read_test PRIVATE include)
  add_test(NAME batch_read_test COMMAND batch_read_test)
  # the model's geometry from its metadata, and the check against the kernels' compiled shape (synthetic)
  add_executable(gguf_geometry_test tests/core/gguf_geometry_test.cpp)
  target_link_libraries(gguf_geometry_test PRIVATE strata_artifact)
  add_test(NAME gguf_geometry_test COMMAND gguf_geometry_test)

  # the hipBLASLt tuning header's architecture gate: RDNA4 siblings share solution IDs, RDNA3 siblings share
  # theirs, cross-family pairs and version mismatches stay rejected (synthetic tables: no model, no GPU)
  add_executable(tuning_header_arch_family_test tests/prefill/tuning_header_arch_family_test.cpp)
  target_include_directories(tuning_header_arch_family_test PRIVATE "${CMAKE_CURRENT_SOURCE_DIR}/src/prefill")
  add_test(NAME tuning_header_arch_family_test COMMAND tuning_header_arch_family_test)

  # the auto layer split's minimum stage size (#1616): header-only rule, no GPU
  add_executable(split_rules_test tests/program/split_rules_test.cpp)
  target_include_directories(split_rules_test PRIVATE include)
  add_test(NAME split_rules_test COMMAND split_rules_test)

  # #3797: the out-of-memory report at start (per card free/total, the largest other programs): the text is a pure
  # function; the driver query (NVML, loaded at run time) is only checked to be safe to call. No GPU needed.
  add_executable(gpu_report_test tests/program/gpu_report_test.cpp)
  target_include_directories(gpu_report_test PRIVATE include)
  target_link_libraries(gpu_report_test PRIVATE ${CMAKE_DL_LIBS})
  add_test(NAME gpu_report_test COMMAND gpu_report_test)

  # #3702: the later-stage reserve of the layer split search (header-only arithmetic, pinned against the old expression)
  add_executable(stage_reserve_test tests/program/stage_reserve_test.cpp)
  target_include_directories(stage_reserve_test PRIVATE include)
  add_test(NAME stage_reserve_test COMMAND stage_reserve_test)

  # the layer split's spill walk (header-only arithmetic: who is short, which step comes next)
  add_executable(split_walk_test tests/program/split_walk_test.cpp)
  target_include_directories(split_walk_test PRIVATE include)
  add_test(NAME split_walk_test COMMAND split_walk_test)

  # the prompt planner's host-memory model (#2824): header-only, no GPU
  add_executable(prefill_plan_test tests/program/prefill_plan_test.cpp)
  target_include_directories(prefill_plan_test PRIVATE include)
  add_test(NAME prefill_plan_test COMMAND prefill_plan_test)

  # the default CPU prefill share is not armed when every expert is resident (#1595): header-only rule, no GPU
  add_executable(share_rules_test tests/prefill/share_rules_test.cpp)
  target_include_directories(share_rules_test PRIVATE include)
  add_test(NAME share_rules_test COMMAND share_rules_test)

  # STRATA_SELECT_TWO_PASS: the bound between the single-pass and 3xTF32 QSA block scores and the cut-off rule that
  # keeps the selection (header-only, no GPU)
  add_executable(qsa_two_pass_test tests/prefill/qsa_two_pass_test.cpp)
  target_include_directories(qsa_two_pass_test PRIVATE include)
  add_test(NAME qsa_two_pass_test COMMAND qsa_two_pass_test)
  # STRATA_SELECT_TWO_PASS: the request's parse and the line it prints once where the scorer cannot run (no GPU)
  add_executable(qsa_two_pass_status_test tests/prefill/qsa_two_pass_status_test.cpp)
  target_include_directories(qsa_two_pass_status_test PRIVATE include)
  add_test(NAME qsa_two_pass_status_test COMMAND qsa_two_pass_status_test
           "${CMAKE_CURRENT_BINARY_DIR}/qsa_two_pass_status_test.log")
           
  # STRATA_PREFILL_ROUTED_STREAM: the prompt stream's issuer walk and route gate, on a simulated link and consumer
  # (header-only, no GPU)
  add_executable(routed_stream_test tests/prefill/routed_stream_test.cpp)
  target_include_directories(routed_stream_test PRIVATE include)
  target_link_libraries(routed_stream_test PRIVATE Threads::Threads)
  add_test(NAME routed_stream_test COMMAND routed_stream_test)

  # PR #4006: STRATA_PREFILL_TIMING does not record a mark in the phase already running (header-only, no GPU)
  add_executable(timer_marks_test tests/prefill/timer_marks_test.cpp)
  target_include_directories(timer_marks_test PRIVATE include)
  add_test(NAME timer_marks_test COMMAND timer_marks_test)

  # PR #5108: the MMQ row table's over-read margin follows the backend tile bound; every fixup read stays inside (header-only, no GPU)
  add_executable(mmq_ids_margin_test tests/prefill/mmq_ids_margin_test.cpp)
  target_include_directories(mmq_ids_margin_test PRIVATE include)
  add_test(NAME mmq_ids_margin_test COMMAND mmq_ids_margin_test)

  # PR #4844: the adaptive tier's copy-in range across serial, pipelined and drained rounds (header-only, no GPU)
  add_executable(adaptive_range_test tests/program/adaptive_range_test.cpp)
  target_include_directories(adaptive_range_test PRIVATE include)
  add_test(NAME adaptive_range_test COMMAND adaptive_range_test)

  # P0.1 (lite): STRATA_MANIFEST, the machine-readable record of the selected paths, tensor formats, engine and budgets (header-only, no GPU)
  add_executable(run_manifest_test tests/core/run_manifest_test.cpp)
  target_include_directories(run_manifest_test PRIVATE include)
  add_test(NAME run_manifest_test COMMAND run_manifest_test)

  # the Windows HIP auto prefill chunk cap (#1630) and the commit-limit verdict (#1607): header-only, no GPU
  add_executable(host_limits_test tests/platform/host_limits_test.cpp)
  target_include_directories(host_limits_test PRIVATE include)
  add_test(NAME host_limits_test COMMAND host_limits_test)
  # #4733: an engine started at Low I/O priority goes back to Normal (Windows; else skips)
  add_executable(io_priority_test tests/platform/io_priority_test.cpp src/platform/io_priority.cpp)
  target_include_directories(io_priority_test PRIVATE include)
  add_test(NAME io_priority_test COMMAND io_priority_test)
  set_tests_properties(io_priority_test PROPERTIES SKIP_RETURN_CODE 77)
endif()

# ---- strata_plan ----------------------------------------------------------
# Also header-only.  A "planner" that hides state would be one whose decision is not inspectable, and the
# whole point of P1.S9 is that the decision is arithmetic you can check by hand.
add_library(strata_plan INTERFACE)
target_include_directories(strata_plan INTERFACE ${CMAKE_CURRENT_SOURCE_DIR}/include)
target_link_libraries(strata_plan INTERFACE strata_warnings)

add_executable(strata-plan src/plan/plan_main.cpp)
target_link_libraries(strata-plan PRIVATE strata_plan)

# ---- strata_core (GPU runtime) ---------------------------------------------
# Only built when a GPU backend is on, so the Phase 1 targets keep working without a toolkit.
strata_gpu_engine()

strata_cpu_kernels()

# ---- fork additions to the GPU engine (cmake/strata_gpu_engine.cmake is upstream's, left as it is) -------------------------
# strata_gpu_engine() above has already run in this scope, so the targets it made exist: what follows adds our sources, links and
# test programs to them (second expert server, EP_SPLIT, elastic cache, prefix file, the #4942 interleaved-kernel checks).
if(STRATA_ENABLE_CUDA OR STRATA_ENABLE_HIP)
  # PEER_FUSED (docs/PEER_FUSED.md s.8, lab/upsync-v0.2.9): the persistent expert server of STRATA_PEER_FUSED=1
  # (peer_fused_kernel, included inside iq_kernels.cu) against today's peer path - quantize_q8_1_rows +
  # native_expert_grouped + scatter_rows_f32 - memcmp on the rows, on synthetic blobs, over several steps with empty
  # groups and a host-made doorbell; GPU, no model.  A .cu because its fake primary can be a kernel on a second card.
  # Not in the default build: `cmake --build <dir> --target peer_fused_test`.  CUDA only, like the benches above: the
  # kernel's cooperative launch, mapped-memory polling and system fences are the CUDA runtime's (not on gfx906, whose
  # .cu files compile as HIP: __stwt, __nanosleep and %globaltimer are not there, and peer_fused.cuh is HIP-guarded).
  if(STRATA_ENABLE_CUDA AND NOT STRATA_HIP_GFX906)
    add_executable(peer_fused_test EXCLUDE_FROM_ALL tools/peer_fused_test.cu)
    target_include_directories(peer_fused_test PRIVATE ${CMAKE_CUDA_TOOLKIT_INCLUDE_DIRECTORIES})
    target_link_libraries(peer_fused_test PRIVATE strata_kernels strata_core CUDA::cudart)
  endif()
  # EP_SPLIT_PREFILL (docs/EP_SPLIT_PREFILL.md s.5, s.11, lab/upsync-v0.2.9): the state mover of STRATA_EP_SPLIT_PREFILL=1
  # (ep_split.cpp, the running state of layers [K, n) between the peer stage's session and the primary's full one) bit
  # for bit on sessions carved by session_init, kv_mode 0 and 1, in both directions, plus the Sync bookkeeping and the
  # mid-prompt checkpoint merge; host code only, GPU, no model.  Not in the default build: `cmake --build <dir>
  # --target ep_split_test`.  CUDA only, like peer_fused_test above (the mover's pinned pieces and streams are the CUDA runtime's).
  if(STRATA_ENABLE_CUDA AND NOT STRATA_HIP_GFX906)
    add_executable(ep_split_test EXCLUDE_FROM_ALL tools/ep_split_test.cpp)
    target_include_directories(ep_split_test PRIVATE include ${CMAKE_CUDA_TOOLKIT_INCLUDE_DIRECTORIES})
    target_link_libraries(ep_split_test PRIVATE strata_engine strata_kernels strata_core CUDA::cudart)
  endif()
  # PEER_PREFILL2 (docs/PEER_PREFILL2.md s.6, lab/upsync-v0.2.9): the second expert server's prompt rows of
  # STRATA_PEER2_PREFILL=1 (prefill.cpp's rows-in path: the routed rows' q8_1 bytes over, FP32 rows back through mapped
  # host memory) against the primary's own MMQ path, bit for bit, on synthetic blobs held slot for slot on both cards;
  # GPU, no model.  Links strata_prefill for the MMQ kernels (its public strata_mmq); exit 77 in a build without MMQ.
  # Not in the default build: `cmake --build <dir> --target peer_prefill2_test`.  CUDA only, like ep_split_test above.
  if(STRATA_ENABLE_CUDA AND NOT STRATA_HIP_GFX906)
    add_executable(peer_prefill2_test EXCLUDE_FROM_ALL tools/peer_prefill2_test.cpp)
    target_include_directories(peer_prefill2_test PRIVATE include ${CMAKE_CUDA_TOOLKIT_INCLUDE_DIRECTORIES})
    target_link_libraries(peer_prefill2_test PRIVATE strata_prefill strata_kernels strata_core CUDA::cudart)
  endif()
  # PEER_FUSED s.5: the kernel is built at __launch_bounds__(256, 2) (STRATA_PEER_FUSED_MINB unset or 2); the 3- and
  # 4-blocks-per-SM instances (STRATA_PEER_FUSED_MINB=3|4) are two more large kernels in iq_kernels.cu (one instance
  # holds every type's body), so they exist only in a build that asks for them.  Off (the default) leaves
  # iq_kernels.cu's compile exactly as it is.
  option(STRATA_PEER_FUSED_MINB_ALL "Also build the __launch_bounds__(256, 3) and (256, 4) instances of the STRATA_PEER_FUSED persistent kernel (STRATA_PEER_FUSED_MINB=3|4)" OFF)
  if(STRATA_ENABLE_CUDA AND STRATA_PEER_FUSED_MINB_ALL)
    set_property(SOURCE src/kernels/cuda/iq_kernels.cu APPEND PROPERTY COMPILE_DEFINITIONS STRATA_PEER_FUSED_MINB_ALL)
    message(STATUS "Strata: STRATA_PEER_FUSED_MINB_ALL - the MINB 3 and 4 instances of the peer's persistent kernel are built too")
  endif()
  # EP_SPLIT (ep_split.cpp, the state mover) and the on-disk prefix (--prompt-cache-file) are part of the engine library
  target_sources(strata_engine PRIVATE src/core/ep_split.cpp)
  target_link_libraries(strata_engine PUBLIC strata_prefix_cache)
  if(STRATA_ENABLE_CUDA AND NOT STRATA_HIP_GFX906)
    # --elastic: the automatic elastic expert cache (CUDA virtual memory management through the driver API)
    target_link_libraries(strata_engine PUBLIC CUDA::cuda_driver)
  endif()
  if(STRATA_BUILD_TESTS AND STRATA_NATIVE_EXPERTS AND STRATA_ENABLE_CUDA)
    add_executable(expert_cache_elastic_test tests/core/expert_cache_elastic_test.cpp)
    target_link_libraries(expert_cache_elastic_test PRIVATE strata_engine)
    add_test(NAME expert_cache_elastic_test COMMAND expert_cache_elastic_test)
    set_tests_properties(expert_cache_elastic_test PROPERTIES SKIP_RETURN_CODE 77 TIMEOUT 60)
  endif()
  # fork F4 / #4942: upstream's block defines mmvq_il_parity for CUDA only; HIP gets it here too, with the L2 set-aside shim below
  if(STRATA_ENABLE_HIP AND NOT TARGET mmvq_il_parity)
    add_executable(mmvq_il_parity src/kernels/mmvq_il_parity.cpp)
    target_include_directories(mmvq_il_parity PRIVATE ${_strata_gpu_include_directories})
    target_link_libraries(mmvq_il_parity PRIVATE strata_kernels ${_strata_gpu_runtime_target})
    add_test(NAME mmvq_il_parity COMMAND mmvq_il_parity)
  endif()
  # (c): native_quantize_q8_1_il (one launch) against native_quantize_q8_1 + native_q8_1_interleave, byte for byte.
  add_executable(quantize_il_parity src/kernels/quantize_il_parity.cpp)
  target_include_directories(quantize_il_parity PRIVATE ${_strata_gpu_include_directories})
  target_link_libraries(quantize_il_parity PRIVATE strata_kernels ${_strata_gpu_runtime_target})
  add_test(NAME quantize_il_parity COMMAND quantize_il_parity)
    if(STRATA_ENABLE_HIP)
      set(_strata_il_parity_hip_l2 "${CMAKE_CURRENT_BINARY_DIR}/mmvq_il_parity_hip_l2.h")
      # mmvq_il_parity's --persist and --ab arms read NVIDIA's L2 set-aside APIs, which hip_compat does not map and HIP
      # has no forms of (hipLimit_t has no persisting-L2 limit, hipDeviceAttribute_t no access-policy attributes), so the
      # file's own #if !defined(STRATA_HIP_GFX906) gates are not enough for a wave32 AMD build.  Target-scoped shim, the
      # shape of dsv41's hip_extra.h -include above: HIP's own names where it has them, ids HIP answers as invalid /
      # unsupported where it has none (the two arms stay NVIDIA-only in practice; --bench --emit-table uses neither).
      file(GENERATE OUTPUT "${_strata_il_parity_hip_l2}" CONTENT "
// generated by CMakeLists.txt for mmvq_il_parity on HIP (the L2 set-aside names its --persist / --ab arms call).
#pragma once
#include <hip/hip_runtime.h>
#ifndef cudaDevAttrL2CacheSize
#define cudaDevAttrL2CacheSize hipDeviceAttributeL2CacheSize
#endif
#ifndef cudaDeviceSetLimit
#define cudaDeviceSetLimit hipDeviceSetLimit
#endif
#ifndef cudaDeviceGetLimit
#define cudaDeviceGetLimit hipDeviceGetLimit
#endif
#ifndef cudaStreamSetAttribute
#define cudaStreamSetAttribute hipStreamSetAttribute
#endif
#ifndef cudaStreamAttributeAccessPolicyWindow
#define cudaStreamAttributeAccessPolicyWindow hipStreamAttributeAccessPolicyWindow
#endif
#ifndef cudaStreamAttrValue
#define cudaStreamAttrValue hipStreamAttrValue
#endif
#ifndef cudaAccessPropertyPersisting
#define cudaAccessPropertyPersisting hipAccessPropertyPersisting
#endif
#ifndef cudaAccessPropertyStreaming
#define cudaAccessPropertyStreaming hipAccessPropertyStreaming
#endif
// HIP has no persisting-L2 limit and no access-policy attributes: ids its runtime answers as an unsupported limit and as
// invalid attributes (measured with ilbench's cc_probe: hipErrorUnsupportedLimit, hipErrorInvalidValue).
#ifndef cudaDevAttrMaxPersistingL2CacheSize
#define cudaDevAttrMaxPersistingL2CacheSize ((hipDeviceAttribute_t) -1)
#endif
#ifndef cudaDevAttrMaxAccessPolicyWindowSize
#define cudaDevAttrMaxAccessPolicyWindowSize ((hipDeviceAttribute_t) -1)
#endif
#ifndef cudaLimitPersistingL2CacheSize
#define cudaLimitPersistingL2CacheSize ((hipLimit_t) 0x03)
#endif
")
      target_compile_options(mmvq_il_parity PRIVATE
        "$<$<COMPILE_LANGUAGE:CXX>:SHELL:-include \"${_strata_il_parity_hip_l2}\">")
    endif()
endif()

# ---- plan v0.3 P6: the i-quant models' CPU experts (IQ2_XS / IQ3_XXS files) ----
#
# The CPU computes a native expert with ggml-cpu's own dot products (`ggml_get_type_traits_cpu`), so an IQ
# weight means exactly what it means in llama.cpp.  ggml is built static from the tree vendored in
# third_party/strata-ggml (frozen at the llama.cpp commit in third_party/ggml/VERSION.txt, see its UPSTREAM.md): no download.
# -DSTRATA_GGML_DIR=<llama.cpp checkout> still builds from that checkout instead (an override kept for two releases).
strata_ggml_resolve("${CMAKE_CURRENT_SOURCE_DIR}")
if(STRATA_PREFILL_MMQ AND NOT STRATA_NATIVE_EXPERTS)
  message(FATAL_ERROR "STRATA_PREFILL_MMQ requires STRATA_NATIVE_EXPERTS=ON for GGML sources")
endif()
if(STRATA_ORCA_Q4KS_MMQ AND NOT STRATA_NATIVE_EXPERTS)
  message(FATAL_ERROR "STRATA_ORCA_Q4KS_MMQ requires STRATA_NATIVE_EXPERTS=ON for GGML sources")
endif()
if(STRATA_NATIVE_EXPERTS)
  enable_language(C)
  strata_ggml_configure_root()
  strata_cpu_native_experts()
  strata_ggml_mmq_types()
  # #296: OrcaRouter Q4_K_S's Q5_0 experts (Q4_K and Q5_1 come with STRATA_MMQ_KQUANTS on CUDA, S5's branch)
  if(STRATA_ORCA_Q4KS_MMQ)
    list(APPEND _strata_mmq_types q5_0)
  endif()
  if(STRATA_ENABLE_CUDA)
    strata_cuda_mmq()
  elseif(STRATA_PREFILL_MMQ)
    strata_hip_mmq()
  endif()
  if(STRATA_ENABLE_HIP)
    # Optional real-model diagnostic: pass the GGUF shard containing the native PLE key.
    add_executable(hip_ple_iq4 tests/hip/ple_iq4.cpp)
    target_link_libraries(hip_ple_iq4 PRIVATE strata_kernels strata_hip_runtime ggml-base)
  endif()
  # the NVFP4 AVX-512 and AVX2 rows vs ggml-cpu's vec_dot, and their speed (CPU only; optional arg: a pack's experts.bin)
  add_executable(nvfp4_avx512_parity src/kernels/cpu/nvfp4_avx512_parity.cpp)
  target_link_libraries(nvfp4_avx512_parity PRIVATE strata_kernels_cpu)
  add_test(NAME nvfp4_cpu_parity COMMAND nvfp4_avx512_parity)   # random blocks; the AVX2 rows bit-equal to ggml's
  set_tests_properties(nvfp4_cpu_parity PROPERTIES TIMEOUT 120)
  if(STRATA_ENABLE_CUDA)   # the NVFP4 decode kernels vs ggml's reference
    add_executable(nvfp4_expert_gpu_parity src/kernels/nvfp4_expert_gpu_parity.cpp)
    target_link_libraries(nvfp4_expert_gpu_parity PRIVATE strata_kernels ggml-base CUDA::cudart)
    # the decode experts' weight bandwidth on random blobs of a format pair, and an output hash (bit-for-bit checks)
    add_executable(nvfp4_decode_bench src/kernels/nvfp4_decode_bench.cpp)
    target_link_libraries(nvfp4_decode_bench PRIVATE strata_kernels CUDA::cudart)
  endif()
  if(STRATA_ENABLE_CUDA OR STRATA_ENABLE_HIP)
    add_executable(native_expert_parity src/kernels/native_expert_parity.cpp)
    target_include_directories(native_expert_parity PRIVATE ${_strata_gpu_include_directories})
    # strata_engine: the --load-check mode runs the engine's start-up checks (NativeDense, check_experts_gguf)
    target_link_libraries(native_expert_parity PRIVATE strata_engine strata_kernels strata_kernels_cpu strata_artifact ${_strata_gpu_runtime_target})
    add_executable(native_expert_bench src/kernels/native_expert_bench.cpp)
    target_include_directories(native_expert_bench PRIVATE ${_strata_gpu_include_directories})
    target_link_libraries(native_expert_bench PRIVATE strata_kernels strata_kernels_cpu strata_artifact ${_strata_gpu_runtime_target})
    if(STRATA_ENABLE_CUDA AND STRATA_EXPERIMENTAL_SM60)
      # STRATA_EXPERT_NIBBLE (include/strata/core/expert_nibble.hpp): the 4-bit gate/up expert slots against the
      # production kernels and the GGUF blobs, bit for bit, on a real model's layers (needs a GP100); --header: the
      # contract's host half, no GPU
      add_executable(iq_nibble_parity src/kernels/iq_nibble_parity.cpp)
      target_include_directories(iq_nibble_parity PRIVATE ${_strata_gpu_include_directories})
      target_link_libraries(iq_nibble_parity PRIVATE strata_kernels strata_kernels_cpu strata_artifact ggml-base ${_strata_gpu_runtime_target})
      if(STRATA_BUILD_TESTS)
        add_test(NAME iq_nibble_parity_header COMMAND iq_nibble_parity --header)
        # the synthetic experts of every gate/up x down format pair, no model (skips with 77 on a card that is not a GP100)
        add_test(NAME iq_nibble_parity_synthetic COMMAND iq_nibble_parity --synthetic-only)
        set_tests_properties(iq_nibble_parity_synthetic PROPERTIES SKIP_RETURN_CODE 77 TIMEOUT 900)
      endif()
    endif()
    # MODEL-UPDATE: the dense Q8_0 GEMVs at the engine's shapes against the memory bandwidth (a benchmark, no test)
    add_executable(q8_gemv_bench src/kernels/q8_gemv_bench.cpp)
    target_include_directories(q8_gemv_bench PRIVATE ${_strata_gpu_include_directories})
    target_link_libraries(q8_gemv_bench PRIVATE strata_kernels ${_strata_gpu_runtime_target})
    # EXPDEC (docs/EXPERTDEC.md): the grouped experts' default kernels against STRATA_EXPDEC=1 at T = 1..4, bit for bit
    # and in microseconds, on synthetic blobs read cold from DRAM
    add_executable(expdec_bench EXCLUDE_FROM_ALL src/kernels/expdec_bench.cpp)
    target_include_directories(expdec_bench PRIVATE ${_strata_gpu_include_directories})
    target_link_libraries(expdec_bench PRIVATE strata_kernels ${_strata_gpu_runtime_target})
    if(STRATA_BUILD_TESTS AND STRATA_ENABLE_CUDA)
      # Unsloth UD-Q4_K_XL's three expert pairs (and Q8_0 alone, its embedding) on synthetic ggml-quantized weights
      # at 2560/640: GPU and CPU experts against the float reference, GPU dequantizers against to_float.
      # q4_K/q5_0 and q5_0/q8_0: a plain llama-quantize "_M" tier GGUF (not Unsloth's dynamic method) commonly
      # puts Q5_0 on down-projections (sometimes gate/up too) where this model's native_experts.txt had no GPU
      # kernel for it before; q5_0/q5_0 (both roles Q5_0) is left out of this list since stacking the same
      # lower-precision format on both roles pushes the GPU path over this test's 3% threshold (quantization
      # noise, not a kernel bug -- see the PR).
      # q4_0/q4_0 and q4_0/q4_1: a plain llama-quantize Q4_0 file (bartowski's Qwen3.8-Flash-Next-Q4_0: gate/up Q4_0,
      # down Q4_0 on 42 layers and Q4_1 on 6)
      # q6_K/q8_0: Unsloth UD-Q6_K_XL gate/up Q6_K with Q8_0 down (docs/UNSLOTH_Q6.md). Layer 2 is q8_0/q8_0, already listed.
      set(_strata_parity_pairs q4_K/q5_1 q4_K/q8_0 q5_K/q8_0 q8_0/q8_0 q4_K/q5_0 q5_0/q8_0 q4_0/q4_0 q4_0/q4_1)
      if(STRATA_Q6K_EXPERTS)
        list(APPEND _strata_parity_pairs q6_K/q8_0)
      else()   # a pair without GPU kernels (Q6_K gate/up) is refused by native_expert_supported, not run
        add_test(NAME native_expert_parity_refuses_q6_K COMMAND native_expert_parity --synthetic q6_K/q8_0)
        set_tests_properties(native_expert_parity_refuses_q6_K PROPERTIES WILL_FAIL TRUE)
      endif()
      foreach(_pair ${_strata_parity_pairs})
        string(REPLACE "/" "_" _name "${_pair}")
        add_test(NAME native_expert_parity_${_name} COMMAND native_expert_parity --synthetic ${_pair})
        set_tests_properties(native_expert_parity_${_name} PROPERTIES TIMEOUT 120)
      endforeach()
      add_test(NAME native_expert_parity_q5_1_min COMMAND native_expert_parity --q5_1-min)
      # Unsloth v3.1 Qwen3.8-Flash-Next sets (UD-IQ1_M .. UD-IQ4_XS, round 0214): every gate/up x down pair their layers use,
      # at the real shapes (hidden 2560, FF 640): gate/up IQ2_XXS IQ2_XS IQ2_S IQ3_XXS IQ4_XS (+ IQ1_M, the GSQ-RCO packs'),
      # down Q2_0 IQ4_NL Q5_0 Q8_0.  Q5_0 down (UD-IQ4_XS 33 layers, UD-Q3_K_XL 20) had no test on this model before.
      foreach(_gu iq1_m iq2_xxs iq2_xs iq2_s iq3_xxs iq4_xs)
        foreach(_dn q2_0 iq4_nl q5_0 q8_0)
          add_test(NAME native_expert_parity_v31_${_gu}_${_dn} COMMAND native_expert_parity --synthetic ${_gu}/${_dn})
          set_tests_properties(native_expert_parity_v31_${_gu}_${_dn} PROPERTIES TIMEOUT 180)
        endforeach()
      endforeach()
      # the dense projections (Q5_0, Q6_K, Q5_K, Q4_K, Q8_0, IQ4_XS, IQ4_NL ...): decode MMVQ and the prompt's dequantizers
      add_executable(native_dense_parity src/kernels/native_dense_parity.cpp)
      target_include_directories(native_dense_parity PRIVATE ${_strata_gpu_include_directories})
      target_link_libraries(native_dense_parity PRIVATE strata_kernels ggml-base ${_strata_gpu_runtime_target})
      add_test(NAME native_dense_parity COMMAND native_dense_parity)
      set_tests_properties(native_dense_parity PROPERTIES SKIP_RETURN_CODE 77 TIMEOUT 300)
      # APR: the trellis expert types and the Hadamard activation rotation, when ggml comes from the agention fork
      # (the only tree with these types; marked by its ggml-cuda/tq.cuh)
      if(STRATA_GGML_DIR AND EXISTS "${STRATA_GGML_DIR}/ggml/src/ggml-cuda/tq.cuh")
        set(_strata_tq_pairs tqk6/tqk7 tq2_t/tq2_t)
        if(EXISTS "${STRATA_GGML_DIR}/ggml/src/ggml-cuda/template-instances/mmq-instance-tqp5.cu")   # Gyro-L's pair trellis
          list(APPEND _strata_tq_pairs tqp5/tqp5 tqp6/tqp6 tqp7/tqp7)
        endif()
        foreach(_pair ${_strata_tq_pairs})
          string(REPLACE "/" "_" _name "${_pair}")
          add_test(NAME native_expert_parity_${_name} COMMAND native_expert_parity --synthetic ${_pair})
          add_test(NAME native_expert_parity_hadamard_${_name} COMMAND native_expert_parity --synthetic-hadamard ${_pair})
          set_tests_properties(native_expert_parity_${_name} native_expert_parity_hadamard_${_name} PROPERTIES TIMEOUT 300)
        endforeach()
      endif()
      # #290: the BF16 token embedding (--embd-gguf) gathered and dequantized bit for bit
      add_test(NAME native_expert_parity_bf16_embd COMMAND native_expert_parity --bf16-embd)
    endif()
  endif()
  # Round 0214 (the new Unsloth Qwen3.8-Flash-Next quants UD-IQ1_M .. UD-IQ4_XS), CUDA and HIP: every expert type pair the
  # sets use at 2560/640 (GPU decode kernels, the CPU pool and the prompt path's dequantizers against ggml), and the dense
  # tensor types (Q5_0, Q6_K, the K-quants, IQ4_NL, IQ4_XS, Q2_0) through native_mmvq.
  if(STRATA_BUILD_TESTS AND (STRATA_ENABLE_CUDA OR STRATA_ENABLE_HIP))
    set(_strata_new_gu iq1_m iq2_xxs iq2_xs iq2_s q2_0 iq3_xxs iq4_xs)
    set(_strata_new_dn q2_0 iq4_nl q5_0 q8_0)
    foreach(_g ${_strata_new_gu})
      foreach(_d ${_strata_new_dn})
        add_test(NAME native_expert_parity_new_${_g}_${_d} COMMAND native_expert_parity --synthetic ${_g}/${_d})
        set_tests_properties(native_expert_parity_new_${_g}_${_d} PROPERTIES TIMEOUT 300)
      endforeach()
    endforeach()
    add_executable(native_dense_types_parity src/kernels/native_dense_types_parity.cpp)
    target_include_directories(native_dense_types_parity PRIVATE ${_strata_gpu_include_directories})
    target_link_libraries(native_dense_types_parity PRIVATE strata_kernels strata_kernels_cpu ${_strata_gpu_runtime_target})
    add_test(NAME native_dense_types_parity COMMAND native_dense_types_parity)
    set_tests_properties(native_dense_types_parity PROPERTIES SKIP_RETURN_CODE 77 TIMEOUT 300)
  endif()
  # #296's real-weight checks (OrcaRouter Q4_K_S files; docs/ORCA_Q4_K_S.md) - test programs, not part of a release
  if(STRATA_ENABLE_CUDA AND STRATA_BUILD_TESTS)
    add_executable(ple_q5_parity src/kernels/ple_q5_parity.cpp)
    target_link_libraries(ple_q5_parity PRIVATE strata_kernels strata_artifact ggml-base ${_strata_gpu_runtime_target})
    add_executable(q5_projection_parity src/kernels/q5_projection_parity.cpp)
    target_include_directories(q5_projection_parity PRIVATE ${_strata_gpu_include_directories})
    target_link_libraries(q5_projection_parity PRIVATE strata_kernels strata_artifact ggml-base ${_strata_gpu_runtime_target})
    # #291: the FP8 PLE table (tools/ple_fp8_pack.py) through PleTable vs the checkpoint's own values
    add_executable(ple_fp8_parity src/kernels/ple_fp8_parity.cpp)
    target_link_libraries(ple_fp8_parity PRIVATE strata_kernels ${_strata_gpu_runtime_target})
  endif()
endif()

if(STRATA_ENABLE_CUDA AND STRATA_BUILD_TESTS AND EXISTS "${CMAKE_CURRENT_SOURCE_DIR}/bench/micro/hit_cpu_order_parity.cu")
  add_executable(hit_cpu_order_parity bench/micro/hit_cpu_order_parity.cu)
  target_link_libraries(hit_cpu_order_parity PRIVATE strata_kernels strata_kernels_cpu ${_strata_gpu_runtime_target})
endif()

strata_cpu_tests()

# CUDA release kernels must not spill or keep a per-thread array in local memory: every access there is a round trip
# to device memory in the hottest loops the engine has, and nothing on the build machine says so without -Xptxas -v
# or a GPU.  tools/check_cuda_spills.py reads the resource usage the compiler wrote into the kernel libraries
# (cuobjdump: STACK and LOCAL per entry function and per sm_) and fails on any kernel outside its ALLOWED table, which
# carries the measured allowance of the kernels that hold such an array on purpose.  CUDA only: HIP has no cuobjdump.
strata_cuda_spill_guard()

if(STRATA_BUILD_TESTS)
  enable_testing()
  # Model-level checks may be absent from source distributions.
  if(EXISTS "${CMAKE_CURRENT_SOURCE_DIR}/tests/CMakeLists.txt")
    add_subdirectory(tests)
  endif()
endif()

# ---- HIP gfx906: every CUDA source of the tree (and the ggml-cuda MMQ sources) is compiled as HIP ----
strata_hip_gfx906_sources()

# ---- DeepSeek-V4.1 on a box with no GPU toolkit ------------------------------------------------
# The whole DSV41 block above lives inside if(STRATA_ENABLE_CUDA OR STRATA_ENABLE_HIP), so with neither backend on
# -DSTRATA_BUILD_DSV41=ON was silently unused and the host-side benchmark could not be built at all (found on p1004,
# 2026-10-23: the CPU lane is exactly what a box without a usable GPU needs to measure).  This branch builds the one target
# that has no device code, so the CPU-lane A/B runs anywhere.
strata_cpu_dsv41_host_bench()

# ---- strata-split: the opt-in engine that splits every layer across 2-4 NVIDIA cards (src/split, from sFNX, MIT, John Fink) ----
# OFF by default, Linux + CUDA or HIP only, no other target depends on it; see src/split/CMakeLists.txt and docs/SPLIT_ENGINE.md.
option(STRATA_BUILD_SPLIT "Build strata-split: sFNX's split-every-layer engine, host-memory exchange (Linux, CUDA or HIP gfx1201, 2-4 cards)" OFF)
if(STRATA_BUILD_SPLIT)
  add_subdirectory(src/split)
endif()

# What this configure registered, so a missing capability is a stated skip (tools/check prints this line too).
if(STRATA_BUILD_TESTS)
  set(_strata_test_kinds "cpu")
  if(STRATA_ENABLE_CUDA)
    string(APPEND _strata_test_kinds ", cuda")
  elseif(STRATA_ENABLE_HIP)
    string(APPEND _strata_test_kinds ", hip")
  else()
    string(APPEND _strata_test_kinds " (gpu tests not registered: neither STRATA_ENABLE_CUDA nor STRATA_ENABLE_HIP is on)")
  endif()
  if(NOT STRATA_NATIVE_EXPERTS)
    string(APPEND _strata_test_kinds " (native-expert tests not registered: STRATA_NATIVE_EXPERTS=OFF)")
  endif()
  message(STATUS "strata tests: ON - ${_strata_test_kinds}")
else()
  message(STATUS "strata tests: OFF (STRATA_BUILD_TESTS=OFF)")
endif()
