# Distributed multi-GPU support module for mumax+
#
# Requires:
#   - MPI (with CUDA-aware support recommended)
#   - HeFFTe (for distributed FFT)
#
# Build:
#   cmake -DENABLE_DISTRIBUTED=ON -DHEFFTE_DIR=/path/to/heffte ..
#
# Optional environment variables:
#   MUMAX_CUDA_AWARE_MPI=1  Force CUDA-aware MPI detection

# Find MPI
find_package(MPI REQUIRED)

# Find HeFFTe (optional, for distributed FFT)
find_package(Heffte QUIET
    HINTS ${HEFFTE_DIR} $ENV{HEFFTE_DIR} $ENV{HOME}/heffte
    PATH_SUFFIXES lib/cmake/Heffte share/Heffte
)

if(Heffte_FOUND)
    message(STATUS "HeFFTe found: ${Heffte_DIR}")
    set(HAVE_HEFFTE TRUE)
else()
    message(STATUS "HeFFTe not found - distributed FFT will not be available")
    set(HAVE_HEFFTE FALSE)
endif()

# CUDA 11.x + GCC 11 compatibility workarounds
if(CMAKE_CUDA_COMPILER_VERSION VERSION_LESS "12.0")
    # Check for older GCC versions that work better with CUDA 11.x
    find_program(GCC10_CXX g++-10)
    find_program(GCC9_CXX g++-9)

    if(GCC10_CXX)
        message(STATUS "Using g++-10 as CUDA host compiler for CUDA 11.x compatibility")
        set(CMAKE_CUDA_FLAGS "${CMAKE_CUDA_FLAGS} -ccbin ${GCC10_CXX}")
    elseif(GCC9_CXX)
        message(STATUS "Using g++-9 as CUDA host compiler for CUDA 11.x compatibility")
        set(CMAKE_CUDA_FLAGS "${CMAKE_CUDA_FLAGS} -ccbin ${GCC9_CXX}")
    endif()
endif()

# Define the distributed library
add_library(distributed STATIC
    mpicontext.hpp
    mpicontext.cu
    distributedgrid.hpp
    distributedgrid.cu
    haloexchanger.hpp
    haloexchanger.cu
)

# Include directories
target_include_directories(distributed PRIVATE
    ${CMAKE_CUDA_TOOLKIT_INCLUDE_DIRECTORIES}
    ${MPI_CXX_INCLUDE_DIRS}
)
target_include_directories(distributed PUBLIC
    ${CMAKE_SOURCE_DIR}/src
    .
)

# Link libraries
target_link_libraries(distributed PUBLIC
    ${MPI_CXX_LIBRARIES}
)

# Link HeFFTe if available
if(HAVE_HEFFTE)
    target_link_libraries(distributed PUBLIC Heffte::Heffte)
    target_compile_definitions(distributed PUBLIC MUMAX_HAVE_HEFFTE)
endif()

# MPI compile flags
target_compile_options(distributed PRIVATE ${MPI_CXX_COMPILE_FLAGS})

# Export compile definitions for CUDA-aware MPI detection
# These are checked at runtime but also useful for conditional compilation
target_compile_definitions(distributed PRIVATE
    $<$<BOOL:${MPI_CXX_LINK_FLAGS}>:MPI_LINK_FLAGS="${MPI_CXX_LINK_FLAGS}">
)

# Infrastructure test
add_executable(test_infrastructure test_infrastructure.cu)
target_link_libraries(test_infrastructure PRIVATE distributed)
target_include_directories(test_infrastructure PRIVATE
    ${CMAKE_CUDA_TOOLKIT_INCLUDE_DIRECTORIES}
    ${MPI_CXX_INCLUDE_DIRS}
)
