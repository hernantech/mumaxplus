#pragma once

/**
 * HaloExchanger - MPI halo exchange service for distributed fields
 *
 * Handles the exchange of ghost cell data between neighboring ranks
 * in the Z-slab decomposition. Uses CUDA-aware MPI for direct GPU
 * memory transfers when available.
 *
 * Exchange pattern (for each field component):
 *   Rank N sends upper boundary -> Rank N+1's lower halo
 *   Rank N sends lower boundary -> Rank N-1's upper halo
 *
 * Supports both blocking (synchronous) and non-blocking (asynchronous)
 * exchange modes.
 */

#include "distributedgrid.hpp"
#include "mpicontext.hpp"  // Includes mpi.h with C++ bindings disabled
#include "core/datatypes.hpp"

#include <vector>

namespace mumaxplus {

/**
 * HaloExchanger class.
 *
 * Manages MPI halo exchange for multi-component fields stored in
 * GPU buffers with Z-slab decomposition.
 */
class HaloExchanger {
public:
    /**
     * Construct a HaloExchanger for the given grid.
     *
     * @param grid The distributed grid configuration
     */
    explicit HaloExchanger(const DistributedGrid& grid);

    /**
     * Destructor - cleans up any pending requests.
     */
    ~HaloExchanger();

    // Disable copy (MPI requests are not copyable)
    HaloExchanger(const HaloExchanger&) = delete;
    HaloExchanger& operator=(const HaloExchanger&) = delete;

    // Enable move
    HaloExchanger(HaloExchanger&&) noexcept;
    HaloExchanger& operator=(HaloExchanger&&) noexcept;

    // ========================================================================
    // Synchronous exchange
    // ========================================================================

    /**
     * Perform a blocking halo exchange for a single-component field.
     *
     * @param data Pointer to GPU buffer (padded layout with halos)
     */
    void exchange(real* data);

    /**
     * Perform a blocking halo exchange for a multi-component field.
     *
     * @param components Vector of GPU buffer pointers (one per component)
     */
    void exchange(const std::vector<real*>& components);

    // ========================================================================
    // Asynchronous exchange
    // ========================================================================

    /**
     * Begin a non-blocking halo exchange for a single-component field.
     * Call waitExchange() to complete the operation.
     *
     * @param data Pointer to GPU buffer (padded layout with halos)
     */
    void beginExchange(real* data);

    /**
     * Begin a non-blocking halo exchange for a multi-component field.
     * Call waitExchange() to complete the operation.
     *
     * @param components Vector of GPU buffer pointers (one per component)
     */
    void beginExchange(const std::vector<real*>& components);

    /**
     * Wait for all pending non-blocking exchanges to complete.
     */
    void waitExchange();

    /**
     * Check if there are pending non-blocking exchanges.
     */
    bool hasPendingExchange() const { return !requests_.empty(); }

    // ========================================================================
    // Configuration
    // ========================================================================

    /**
     * Get the grid configuration.
     */
    const DistributedGrid& grid() const { return grid_; }

    /**
     * Check if CUDA-aware MPI is being used.
     */
    bool isCudaAwareMPI() const { return cudaAwareMPI_; }

private:
    /**
     * Internal: Exchange halos for one component.
     *
     * @param data GPU buffer pointer
     * @param async If true, use non-blocking MPI calls
     */
    void exchangeComponent(real* data, bool async);

    /**
     * Detect if MPI implementation supports CUDA-aware operations.
     */
    static bool detectCudaAwareMPI();

    DistributedGrid grid_;
    bool cudaAwareMPI_;

    // Staging buffers for non-CUDA-aware MPI
    std::vector<real> sendLowerHost_;
    std::vector<real> sendUpperHost_;
    std::vector<real> recvLowerHost_;
    std::vector<real> recvUpperHost_;

    // Pending non-blocking requests
    std::vector<MPI_Request> requests_;
    std::vector<real*> pendingBuffers_;  // Buffers involved in pending exchange
};

}  // namespace mumaxplus
