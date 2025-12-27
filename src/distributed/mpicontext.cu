#include "mpicontext.hpp"

#include <mpi.h>
#include <cuda_runtime.h>

#include <iostream>
#include <stdexcept>
#include <cstdlib>

namespace mumaxplus {

// ============================================================================
// Module-level static state (same pattern as cudastream.cu)
// ============================================================================

static bool s_initialized = false;
static int s_rank = 0;
static int s_size = 1;
static int s_deviceId = 0;

// Z-slab decomposition state
static int s_globalNz = 0;
static int s_localNz = 0;
static int s_zStart = 0;
static int s_zEnd = 0;
static int s_lowerNeighbor = -1;
static int s_upperNeighbor = -1;

static MPI_Comm s_comm = MPI_COMM_WORLD;

// ============================================================================
// MPIContext implementation
// ============================================================================

void MPIContext::initialize(int& argc, char**& argv) {
    if (s_initialized) {
        return;  // Already initialized, no-op
    }

    // Initialize MPI
    int provided;
    MPI_Init_thread(&argc, &argv, MPI_THREAD_FUNNELED, &provided);

    if (provided < MPI_THREAD_FUNNELED) {
        std::cerr << "Warning: MPI does not support MPI_THREAD_FUNNELED\n";
    }

    MPI_Comm_rank(MPI_COMM_WORLD, &s_rank);
    MPI_Comm_size(MPI_COMM_WORLD, &s_size);

    // Bind GPU: round-robin assignment based on rank and local device count
    int deviceCount = 0;
    cudaError_t err = cudaGetDeviceCount(&deviceCount);

    if (err != cudaSuccess || deviceCount == 0) {
        std::cerr << "Rank " << s_rank << ": No CUDA devices found!\n";
        MPI_Abort(MPI_COMM_WORLD, 1);
    }

    // Round-robin GPU binding (works for multi-node with same GPU count per node)
    s_deviceId = s_rank % deviceCount;
    err = cudaSetDevice(s_deviceId);

    if (err != cudaSuccess) {
        std::cerr << "Rank " << s_rank << ": Failed to set CUDA device "
                  << s_deviceId << "\n";
        MPI_Abort(MPI_COMM_WORLD, 1);
    }

    s_comm = MPI_COMM_WORLD;
    s_initialized = true;

    // Print initialization info from rank 0
    if (s_rank == 0) {
        std::cout << "MPIContext: Initialized with " << s_size << " ranks, "
                  << deviceCount << " GPUs per node\n";
    }

    // Synchronize to ensure all ranks are ready
    MPI_Barrier(MPI_COMM_WORLD);
}

void MPIContext::configureDecomposition(int globalNz) {
    if (!s_initialized) {
        throw std::runtime_error("MPIContext::configureDecomposition called before initialize()");
    }

    if (globalNz < s_size) {
        if (s_rank == 0) {
            std::cerr << "Error: globalNz (" << globalNz << ") < MPI size ("
                      << s_size << "). Need at least 1 Z-plane per rank.\n";
        }
        MPI_Abort(MPI_COMM_WORLD, 1);
    }

    s_globalNz = globalNz;

    // Compute Z-slab decomposition
    // Distribute Z-planes as evenly as possible
    int baseNz = globalNz / s_size;
    int remainder = globalNz % s_size;

    // Ranks 0..remainder-1 get one extra Z-plane
    if (s_rank < remainder) {
        s_localNz = baseNz + 1;
        s_zStart = s_rank * (baseNz + 1);
    } else {
        s_localNz = baseNz;
        s_zStart = remainder * (baseNz + 1) + (s_rank - remainder) * baseNz;
    }
    s_zEnd = s_zStart + s_localNz - 1;

    // Determine neighbors
    s_lowerNeighbor = (s_rank > 0) ? s_rank - 1 : -1;
    s_upperNeighbor = (s_rank < s_size - 1) ? s_rank + 1 : -1;

    // Print decomposition info
    if (s_rank == 0) {
        std::cout << "MPIContext: Z-slab decomposition for Nz=" << globalNz << ":\n";
    }
    MPI_Barrier(MPI_COMM_WORLD);

    for (int r = 0; r < s_size; r++) {
        if (r == s_rank) {
            std::cout << "  Rank " << s_rank << ": GPU " << s_deviceId
                      << ", Z=[" << s_zStart << "," << s_zEnd << "] ("
                      << s_localNz << " planes)\n";
            std::cout.flush();
        }
        MPI_Barrier(MPI_COMM_WORLD);
    }
}

void MPIContext::finalize() {
    if (!s_initialized) {
        return;
    }

    MPI_Barrier(MPI_COMM_WORLD);
    MPI_Finalize();

    s_initialized = false;
    s_rank = 0;
    s_size = 1;
    s_globalNz = 0;
    s_localNz = 0;
    s_zStart = 0;
    s_zEnd = 0;
    s_lowerNeighbor = -1;
    s_upperNeighbor = -1;
}

bool MPIContext::isInitialized() {
    return s_initialized;
}

int MPIContext::rank() {
    return s_rank;
}

int MPIContext::size() {
    return s_size;
}

int MPIContext::deviceId() {
    return s_deviceId;
}

int MPIContext::globalNz() {
    return s_globalNz;
}

int MPIContext::localNz() {
    return s_localNz;
}

int MPIContext::zStart() {
    return s_zStart;
}

int MPIContext::zEnd() {
    return s_zEnd;
}

int MPIContext::lowerNeighbor() {
    return s_lowerNeighbor;
}

int MPIContext::upperNeighbor() {
    return s_upperNeighbor;
}

void MPIContext::barrier() {
    if (s_initialized) {
        MPI_Barrier(s_comm);
    }
}

void MPIContext::abort(const std::string& msg) {
    std::cerr << "Rank " << s_rank << " ABORT: " << msg << std::endl;
    if (s_initialized) {
        MPI_Abort(s_comm, 1);
    } else {
        std::exit(1);
    }
}

MPI_Comm MPIContext::comm() {
    return s_comm;
}

}  // namespace mumaxplus
