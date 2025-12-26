cmake_minimum_required(VERSION 3.18)

# Set CUDA architectures before project() to avoid auto-detection issues
# 75 = RTX 2070 (Turing), 86 = RTX A5000/3090 (Ampere)
if(NOT DEFINED CMAKE_CUDA_ARCHITECTURES)
    set(CMAKE_CUDA_ARCHITECTURES "75;86")
endif()

project(mumaxplus_distributed_poc LANGUAGES CXX CUDA)

# ============================================================================
# Configuration
# ============================================================================

set(CMAKE_CXX_STANDARD 14)
set(CMAKE_CXX_STANDARD_REQUIRED ON)
set(CMAKE_CUDA_STANDARD 14)
set(CMAKE_CUDA_STANDARD_REQUIRED ON)

# Build type
if(NOT CMAKE_BUILD_TYPE)
    set(CMAKE_BUILD_TYPE Release)
endif()

set(CMAKE_CXX_FLAGS_RELEASE "-O3")
set(CMAKE_CUDA_FLAGS_RELEASE "-O3")

# ============================================================================
# Find Dependencies
# ============================================================================

# MPI
find_package(MPI REQUIRED)
message(STATUS "MPI Found: ${MPI_CXX_COMPILER}")

# CUDA
find_package(CUDAToolkit REQUIRED)
message(STATUS "CUDA Toolkit: ${CUDAToolkit_VERSION}")

# HeFFTe
# Set HEFFTE_DIR if not in standard path:
#   cmake -DHEFFTE_DIR=/path/to/heffte/install ..
if(HEFFTE_DIR)
    list(APPEND CMAKE_PREFIX_PATH "${HEFFTE_DIR}")
    set(Heffte_DIR "${HEFFTE_DIR}/lib/cmake/Heffte")
endif()
find_package(Heffte QUIET)

if(Heffte_FOUND)
    message(STATUS "HeFFTe found via CMake config")
    set(HEFFTE_USE_IMPORTED_TARGET TRUE)
else()
    # Try pkg-config as fallback
    find_package(PkgConfig)
    if(PkgConfig_FOUND)
        pkg_check_modules(HEFFTE heffte)
    endif()

    if(NOT HEFFTE_FOUND)
        message(WARNING "
========================================================================
HeFFTe not found automatically!

To build this PoC, you need HeFFTe installed with CUDA support.

Installation instructions:
  git clone https://github.com/icl-utk-edu/heffte.git
  cd heffte && mkdir build && cd build
  cmake -DCMAKE_INSTALL_PREFIX=/opt/heffte \\
        -DHeffte_ENABLE_CUDA=ON \\
        -DHeffte_ENABLE_MPI=ON \\
        -DCMAKE_CUDA_ARCHITECTURES=70;80;86 ..
  make -j install

Then configure this project with:
  cmake -DHEFFTE_DIR=/opt/heffte ..

Attempting to continue with manual paths...
========================================================================
")
        # Set manual paths - user should override these
        set(HEFFTE_INCLUDE_DIRS "/opt/heffte/include" CACHE PATH "HeFFTe include directory")
        set(HEFFTE_LIBRARY_DIRS "/opt/heffte/lib" CACHE PATH "HeFFTe library directory")
    endif()
endif()

# ============================================================================
# Targets
# ============================================================================

# HeFFTe Proof-of-Concept
add_executable(heffte_poc heffte_poc.cu)

if(HEFFTE_USE_IMPORTED_TARGET)
    target_link_libraries(heffte_poc PRIVATE
        MPI::MPI_CXX
        CUDA::cudart
        CUDA::cufft
        Heffte::Heffte
    )
else()
    target_include_directories(heffte_poc PRIVATE
        ${MPI_CXX_INCLUDE_DIRS}
        ${HEFFTE_INCLUDE_DIRS}
    )
    target_link_directories(heffte_poc PRIVATE
        ${HEFFTE_LIBRARY_DIRS}
    )
    target_link_libraries(heffte_poc PRIVATE
        MPI::MPI_CXX
        CUDA::cudart
        CUDA::cufft
        heffte
    )
endif()

# Set CUDA architectures (adjust for your GPU)
# 75 = RTX 2070, 80 = A100, 86 = RTX 3090/A5000, 89 = RTX 4090
if(CMAKE_CUDA_ARCHITECTURES)
    set_target_properties(heffte_poc PROPERTIES
        CUDA_ARCHITECTURES "${CMAKE_CUDA_ARCHITECTURES}"
    )
else()
    set_target_properties(heffte_poc PROPERTIES
        CUDA_ARCHITECTURES "75;86"
    )
endif()

# ============================================================================
# Validation Test (Multi-GPU vs Single-GPU)
# ============================================================================

add_executable(validation_test validation_test.cu)

if(HEFFTE_USE_IMPORTED_TARGET)
    target_link_libraries(validation_test PRIVATE
        MPI::MPI_CXX
        CUDA::cudart
        CUDA::cufft
        Heffte::Heffte
    )
else()
    target_include_directories(validation_test PRIVATE
        ${MPI_CXX_INCLUDE_DIRS}
        ${HEFFTE_INCLUDE_DIRS}
    )
    target_link_directories(validation_test PRIVATE
        ${HEFFTE_LIBRARY_DIRS}
    )
    target_link_libraries(validation_test PRIVATE
        MPI::MPI_CXX
        CUDA::cudart
        CUDA::cufft
        heffte
    )
endif()

if(CMAKE_CUDA_ARCHITECTURES)
    set_target_properties(validation_test PROPERTIES
        CUDA_ARCHITECTURES "${CMAKE_CUDA_ARCHITECTURES}"
    )
else()
    set_target_properties(validation_test PROPERTIES
        CUDA_ARCHITECTURES "75;86"
    )
endif()

# ============================================================================
# Halo Exchange Test (Phase 2)
# ============================================================================

add_executable(halo_exchange_test halo_exchange_test.cu)

target_include_directories(halo_exchange_test PRIVATE
    ${MPI_CXX_INCLUDE_DIRS}
)

target_link_libraries(halo_exchange_test PRIVATE
    MPI::MPI_CXX
    CUDA::cudart
)

if(CMAKE_CUDA_ARCHITECTURES)
    set_target_properties(halo_exchange_test PROPERTIES
        CUDA_ARCHITECTURES "${CMAKE_CUDA_ARCHITECTURES}"
    )
else()
    set_target_properties(halo_exchange_test PROPERTIES
        CUDA_ARCHITECTURES "75;86"
    )
endif()

# ============================================================================
# Installation
# ============================================================================

install(TARGETS heffte_poc validation_test halo_exchange_test
    RUNTIME DESTINATION bin
)

# ============================================================================
# Testing
# ============================================================================

enable_testing()

# Test with 2 ranks
add_test(NAME heffte_poc_2gpu
    COMMAND ${MPIEXEC_EXECUTABLE} ${MPIEXEC_NUMPROC_FLAG} 2 $<TARGET_FILE:heffte_poc>
)

# Test with 4 ranks (if available)
add_test(NAME heffte_poc_4gpu
    COMMAND ${MPIEXEC_EXECUTABLE} ${MPIEXEC_NUMPROC_FLAG} 4 $<TARGET_FILE:heffte_poc>
)

# Validation test: Multi-GPU vs Single-GPU comparison
add_test(NAME validation_test_2gpu
    COMMAND ${MPIEXEC_EXECUTABLE} ${MPIEXEC_NUMPROC_FLAG} 2 $<TARGET_FILE:validation_test>
)

add_test(NAME validation_test_4gpu
    COMMAND ${MPIEXEC_EXECUTABLE} ${MPIEXEC_NUMPROC_FLAG} 4 $<TARGET_FILE:validation_test>
)

# ============================================================================
# Summary
# ============================================================================

message(STATUS "")
message(STATUS "========================================")
message(STATUS "mumax+ Distributed PoC Configuration")
message(STATUS "========================================")
message(STATUS "Build type: ${CMAKE_BUILD_TYPE}")
message(STATUS "MPI: ${MPI_CXX_COMPILER}")
message(STATUS "CUDA: ${CUDAToolkit_VERSION}")
message(STATUS "HeFFTe: ${HEFFTE_INCLUDE_DIRS}")
message(STATUS "")
message(STATUS "Build with: cmake --build .")
message(STATUS "Test with:  mpirun -np 2 ./heffte_poc")
message(STATUS "Validate:   mpirun -np 2 ./validation_test")
message(STATUS "========================================")
