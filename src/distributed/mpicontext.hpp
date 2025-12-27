#pragma once

/**
 * MPIContext - Singleton for managing MPI state in multi-GPU mumax+
 *
 * Follows the same pattern as getCudaStream() in cudautil/cudastream.cu:
 * module-level static state with accessor functions.
 *
 * Usage:
 *   // In main():
 *   MPIContext::initialize(argc, argv);
 *   MPIContext::configureDecomposition(Nz);
 *   ...
 *   MPIContext::finalize();
 *
 *   // Anywhere else:
 *   if (isMPIDistributed()) {
 *       int myRank = getMPIRank();
 *       ...
 *   }
 */

#include <mpi.h>
#include <string>

namespace mumaxplus {

/**
 * MPIContext singleton class.
 *
 * Manages:
 * - MPI initialization/finalization
 * - GPU device binding (round-robin)
 * - Z-slab domain decomposition parameters
 */
class MPIContext {
public:
    /**
     * Initialize MPI and bind this rank to a GPU.
     * Must be called before any other MPI or CUDA operations.
     *
     * @param argc Reference to main's argc (may be modified by MPI_Init)
     * @param argv Reference to main's argv (may be modified by MPI_Init)
     */
    static void initialize(int& argc, char**& argv);

    /**
     * Configure Z-slab decomposition for a given global Nz.
     * Must be called after initialize() and before any distributed operations.
     *
     * @param globalNz Total number of Z-planes in the simulation
     */
    static void configureDecomposition(int globalNz);

    /**
     * Finalize MPI. Should be called at program exit.
     */
    static void finalize();

    /**
     * Check if MPI has been initialized.
     */
    static bool isInitialized();

    /**
     * Get this process's MPI rank (0 to size-1).
     */
    static int rank();

    /**
     * Get total number of MPI ranks.
     */
    static int size();

    /**
     * Get the GPU device ID bound to this rank.
     */
    static int deviceId();

    /**
     * Get the global Nz (total Z-planes across all ranks).
     */
    static int globalNz();

    /**
     * Get the local Nz (Z-planes owned by this rank).
     */
    static int localNz();

    /**
     * Get the starting Z-index (global) for this rank.
     */
    static int zStart();

    /**
     * Get the ending Z-index (global, inclusive) for this rank.
     */
    static int zEnd();

    /**
     * Get the MPI rank of the lower neighbor (rank-1), or -1 if none.
     */
    static int lowerNeighbor();

    /**
     * Get the MPI rank of the upper neighbor (rank+1), or -1 if none.
     */
    static int upperNeighbor();

    /**
     * Synchronize all MPI ranks (barrier).
     */
    static void barrier();

    /**
     * Abort all MPI ranks with an error message.
     *
     * @param msg Error message to print before aborting
     */
    static void abort(const std::string& msg);

    /**
     * Get the MPI communicator.
     */
    static MPI_Comm comm();

private:
    // Prevent instantiation
    MPIContext() = delete;
    ~MPIContext() = delete;
    MPIContext(const MPIContext&) = delete;
    MPIContext& operator=(const MPIContext&) = delete;
};

// ============================================================================
// Convenience functions (preferred API)
// ============================================================================

/**
 * Get this process's MPI rank.
 * Returns 0 if MPI is not initialized.
 */
inline int getMPIRank() {
    return MPIContext::isInitialized() ? MPIContext::rank() : 0;
}

/**
 * Get total number of MPI ranks.
 * Returns 1 if MPI is not initialized.
 */
inline int getMPISize() {
    return MPIContext::isInitialized() ? MPIContext::size() : 1;
}

/**
 * Check if running in distributed mode (more than 1 MPI rank).
 */
inline bool isMPIDistributed() {
    return MPIContext::isInitialized() && MPIContext::size() > 1;
}

/**
 * Check if this is the root rank (rank 0).
 */
inline bool isMPIRoot() {
    return getMPIRank() == 0;
}

}  // namespace mumaxplus
