#include "distributedgrid.hpp"
#include "mpicontext.hpp"

#include <stdexcept>

namespace mumaxplus {

DistributedGrid::DistributedGrid()
    : globalSize_{0, 0, 0}
    , cellSize_{0.0f, 0.0f, 0.0f}
    , haloWidth_(0)
    , localSize_{0, 0, 0}
    , paddedSize_{0, 0, 0}
    , zStart_(0)
    , zEnd_(0)
    , hasLowerNeighbor_(false)
    , hasUpperNeighbor_(false) {
}

DistributedGrid::DistributedGrid(int3 globalSize, real3 cellSize, int haloWidth)
    : globalSize_(globalSize)
    , cellSize_(cellSize)
    , haloWidth_(haloWidth) {

    if (!MPIContext::isInitialized()) {
        throw std::runtime_error("DistributedGrid: MPIContext not initialized");
    }

    // Configure the decomposition if not already done
    if (MPIContext::globalNz() == 0) {
        MPIContext::configureDecomposition(globalSize.z);
    } else if (MPIContext::globalNz() != globalSize.z) {
        throw std::runtime_error("DistributedGrid: globalNz mismatch with MPIContext");
    }

    // Get decomposition info from MPIContext
    zStart_ = MPIContext::zStart();
    zEnd_ = MPIContext::zEnd();
    int localNz = MPIContext::localNz();

    // Determine neighbors
    hasLowerNeighbor_ = (MPIContext::lowerNeighbor() >= 0);
    hasUpperNeighbor_ = (MPIContext::upperNeighbor() >= 0);

    // Local size is the actual data this rank owns (no halos)
    localSize_ = int3{globalSize.x, globalSize.y, localNz};

    // Padded size includes halos on sides that have neighbors
    int lowerPad = hasLowerNeighbor_ ? haloWidth : 0;
    int upperPad = hasUpperNeighbor_ ? haloWidth : 0;
    paddedSize_ = int3{globalSize.x, globalSize.y, localNz + lowerPad + upperPad};
}

int DistributedGrid::globalToLocalZ(int globalZ) const {
    if (globalZ < zStart_ || globalZ > zEnd_) {
        return -1;  // Not owned by this rank
    }

    // Account for lower halo offset
    int offset = hasLowerNeighbor_ ? haloWidth_ : 0;
    return (globalZ - zStart_) + offset;
}

int DistributedGrid::localToGlobalZ(int localZ) const {
    int offset = hasLowerNeighbor_ ? haloWidth_ : 0;

    // Check if in lower halo
    if (localZ < offset) {
        return -1;  // In halo region
    }

    // Check if in upper halo
    int realEnd = offset + localSize_.z;
    if (localZ >= realEnd) {
        return -1;  // In halo region
    }

    return zStart_ + (localZ - offset);
}

bool DistributedGrid::ownsGlobalZ(int globalZ) const {
    return (globalZ >= zStart_ && globalZ <= zEnd_);
}

int DistributedGrid::paddedIndex(int x, int y, int localZ) const {
    // Row-major: x + y*Nx + z*Nx*Ny
    return x + y * paddedSize_.x + localZ * paddedSize_.x * paddedSize_.y;
}

size_t DistributedGrid::lowerHaloBytes() const {
    if (!hasLowerNeighbor_) return 0;
    return static_cast<size_t>(haloWidth_) * planeSize() * sizeof(real);
}

size_t DistributedGrid::upperHaloBytes() const {
    if (!hasUpperNeighbor_) return 0;
    return static_cast<size_t>(haloWidth_) * planeSize() * sizeof(real);
}

size_t DistributedGrid::boundaryPlaneBytes() const {
    return static_cast<size_t>(haloWidth_) * planeSize() * sizeof(real);
}

}  // namespace mumaxplus
