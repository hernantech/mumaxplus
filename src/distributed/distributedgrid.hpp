#pragma once

/**
 * DistributedGrid - Manages Z-slab decomposition grid information
 *
 * This class provides grid information for both the local slab (what this
 * rank owns) and the global simulation domain. It handles the mapping between
 * local and global coordinates for the Z-slab decomposition strategy.
 *
 * Memory layout for each rank:
 *   [lower_halo | real_data | upper_halo]
 *
 * The halo regions are ghost cells filled by neighboring ranks via MPI.
 */

#include "core/datatypes.hpp"

namespace mumaxplus {

/**
 * DistributedGrid class.
 *
 * Encapsulates:
 * - Global grid dimensions (Nx, Ny, Nz_global)
 * - Local slab dimensions with halo padding
 * - Coordinate mapping between local and global indices
 */
class DistributedGrid {
public:
    /**
     * Construct a DistributedGrid for the given global dimensions and halo width.
     *
     * @param globalSize Global grid dimensions (Nx, Ny, Nz)
     * @param cellSize   Cell size in each dimension (dx, dy, dz)
     * @param haloWidth  Number of halo planes on each side (typically 1-2)
     */
    DistributedGrid(int3 globalSize, real3 cellSize, int haloWidth = 1);

    /**
     * Default constructor (creates an uninitialized grid).
     */
    DistributedGrid();

    // ========================================================================
    // Global grid properties
    // ========================================================================

    /** Global grid dimensions (Nx, Ny, Nz_total) */
    int3 globalSize() const { return globalSize_; }

    /** Cell size (dx, dy, dz) */
    real3 cellSize() const { return cellSize_; }

    /** Total number of cells in global grid */
    int globalCellCount() const {
        return globalSize_.x * globalSize_.y * globalSize_.z;
    }

    // ========================================================================
    // Local slab properties (what this rank owns)
    // ========================================================================

    /** Local slab dimensions WITHOUT halos (Nx, Ny, localNz) */
    int3 localSize() const { return localSize_; }

    /** Local slab dimensions WITH halos (Nx, Ny, localNz + 2*haloWidth) */
    int3 paddedSize() const { return paddedSize_; }

    /** Number of halo planes on each side */
    int haloWidth() const { return haloWidth_; }

    /** Total number of cells in local slab (without halos) */
    int localCellCount() const {
        return localSize_.x * localSize_.y * localSize_.z;
    }

    /** Total number of cells in padded slab (with halos) */
    int paddedCellCount() const {
        return paddedSize_.x * paddedSize_.y * paddedSize_.z;
    }

    /** Number of cells in one XY plane */
    int planeSize() const {
        return globalSize_.x * globalSize_.y;
    }

    // ========================================================================
    // Z-slab decomposition info
    // ========================================================================

    /** Starting Z-index in global coordinates */
    int zStart() const { return zStart_; }

    /** Ending Z-index in global coordinates (inclusive) */
    int zEnd() const { return zEnd_; }

    /** Number of Z-planes this rank owns (excluding halos) */
    int localNz() const { return localSize_.z; }

    /** Whether this rank has a lower neighbor (rank-1) */
    bool hasLowerNeighbor() const { return hasLowerNeighbor_; }

    /** Whether this rank has an upper neighbor (rank+1) */
    bool hasUpperNeighbor() const { return hasUpperNeighbor_; }

    // ========================================================================
    // Coordinate mapping
    // ========================================================================

    /**
     * Convert global Z-index to local Z-index (in padded buffer).
     * The local index accounts for the lower halo offset.
     *
     * @param globalZ Global Z-index
     * @return Local Z-index in padded buffer, or -1 if out of range
     */
    int globalToLocalZ(int globalZ) const;

    /**
     * Convert local Z-index (in padded buffer) to global Z-index.
     *
     * @param localZ Local Z-index in padded buffer
     * @return Global Z-index, or -1 if in halo region
     */
    int localToGlobalZ(int localZ) const;

    /**
     * Check if a global Z-index is owned by this rank.
     *
     * @param globalZ Global Z-index
     * @return true if this rank owns the Z-plane
     */
    bool ownsGlobalZ(int globalZ) const;

    /**
     * Get the linear index in the padded buffer for a given (x, y, localZ).
     * Uses row-major ordering: idx = x + y*Nx + localZ*Nx*Ny
     *
     * @param x X-index
     * @param y Y-index
     * @param localZ Local Z-index (in padded buffer)
     * @return Linear index
     */
    int paddedIndex(int x, int y, int localZ) const;

    // ========================================================================
    // Halo region info (for MPI communication)
    // ========================================================================

    /**
     * Get the starting local Z-index of the lower halo region.
     * Returns 0 (halos are at the start of the buffer).
     */
    int lowerHaloStart() const { return 0; }

    /**
     * Get the starting local Z-index of the real data region.
     * This is where data owned by this rank begins.
     */
    int realDataStart() const { return hasLowerNeighbor_ ? haloWidth_ : 0; }

    /**
     * Get the starting local Z-index of the upper halo region.
     */
    int upperHaloStart() const { return realDataStart() + localSize_.z; }

    /**
     * Get the size of the lower halo region (in bytes) for one component.
     */
    size_t lowerHaloBytes() const;

    /**
     * Get the size of the upper halo region (in bytes) for one component.
     */
    size_t upperHaloBytes() const;

    /**
     * Get the size of one boundary plane (in bytes) for one component.
     */
    size_t boundaryPlaneBytes() const;

private:
    int3 globalSize_;       // Global dimensions
    real3 cellSize_;        // Cell size
    int haloWidth_;         // Halo planes per side

    int3 localSize_;        // Local slab size (without halos)
    int3 paddedSize_;       // Local slab size (with halos)

    int zStart_;            // Global Z-start
    int zEnd_;              // Global Z-end (inclusive)

    bool hasLowerNeighbor_; // Has rank-1 neighbor
    bool hasUpperNeighbor_; // Has rank+1 neighbor
};

}  // namespace mumaxplus
