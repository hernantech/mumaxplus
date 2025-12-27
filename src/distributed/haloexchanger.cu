#include "haloexchanger.hpp"
#include "mpicontext.hpp"

#include <mpi.h>
#include <cuda_runtime.h>

#include <algorithm>
#include <cstdlib>
#include <iostream>

namespace mumaxplus {

// MPI tags for halo exchange
static constexpr int TAG_LOWER_TO_UPPER = 100;
static constexpr int TAG_UPPER_TO_LOWER = 101;

// ============================================================================
// Constructor / Destructor
// ============================================================================

HaloExchanger::HaloExchanger(const DistributedGrid& grid)
    : grid_(grid)
    , cudaAwareMPI_(detectCudaAwareMPI()) {

    // Allocate host staging buffers if not using CUDA-aware MPI
    if (!cudaAwareMPI_) {
        size_t planeElements = grid_.planeSize() * grid_.haloWidth();

        if (grid_.hasLowerNeighbor()) {
            sendLowerHost_.resize(planeElements);
            recvLowerHost_.resize(planeElements);
        }
        if (grid_.hasUpperNeighbor()) {
            sendUpperHost_.resize(planeElements);
            recvUpperHost_.resize(planeElements);
        }
    }

    if (isMPIRoot()) {
        std::cout << "HaloExchanger: CUDA-aware MPI: "
                  << (cudaAwareMPI_ ? "yes" : "no (using host staging)")
                  << std::endl;
    }
}

HaloExchanger::~HaloExchanger() {
    // Cancel any pending requests
    if (!requests_.empty()) {
        for (auto& req : requests_) {
            if (req != MPI_REQUEST_NULL) {
                MPI_Cancel(&req);
                MPI_Request_free(&req);
            }
        }
        requests_.clear();
    }
}

HaloExchanger::HaloExchanger(HaloExchanger&& other) noexcept
    : grid_(other.grid_)
    , cudaAwareMPI_(other.cudaAwareMPI_)
    , sendLowerHost_(std::move(other.sendLowerHost_))
    , sendUpperHost_(std::move(other.sendUpperHost_))
    , recvLowerHost_(std::move(other.recvLowerHost_))
    , recvUpperHost_(std::move(other.recvUpperHost_))
    , requests_(std::move(other.requests_))
    , pendingBuffers_(std::move(other.pendingBuffers_)) {
}

HaloExchanger& HaloExchanger::operator=(HaloExchanger&& other) noexcept {
    if (this != &other) {
        grid_ = other.grid_;
        cudaAwareMPI_ = other.cudaAwareMPI_;
        sendLowerHost_ = std::move(other.sendLowerHost_);
        sendUpperHost_ = std::move(other.sendUpperHost_);
        recvLowerHost_ = std::move(other.recvLowerHost_);
        recvUpperHost_ = std::move(other.recvUpperHost_);
        requests_ = std::move(other.requests_);
        pendingBuffers_ = std::move(other.pendingBuffers_);
    }
    return *this;
}

// ============================================================================
// CUDA-aware MPI detection
// ============================================================================

bool HaloExchanger::detectCudaAwareMPI() {
    // Check environment variable first (user override)
    const char* env = std::getenv("MUMAX_CUDA_AWARE_MPI");
    if (env) {
        return (std::string(env) == "1" || std::string(env) == "true");
    }

    // Check for Open MPI CUDA support
    #if defined(MPIX_CUDA_AWARE_SUPPORT) && MPIX_CUDA_AWARE_SUPPORT
        return true;
    #endif

    // Check MVAPICH2 CUDA support
    #if defined(MVAPICH2_VERSION) && defined(MV2_USE_CUDA)
        return true;
    #endif

    // Default: assume not CUDA-aware (safer)
    return false;
}

// ============================================================================
// Synchronous exchange
// ============================================================================

void HaloExchanger::exchange(real* data) {
    exchangeComponent(data, false);
}

void HaloExchanger::exchange(const std::vector<real*>& components) {
    for (real* comp : components) {
        exchangeComponent(comp, false);
    }
}

// ============================================================================
// Asynchronous exchange
// ============================================================================

void HaloExchanger::beginExchange(real* data) {
    exchangeComponent(data, true);
    pendingBuffers_.push_back(data);
}

void HaloExchanger::beginExchange(const std::vector<real*>& components) {
    for (real* comp : components) {
        exchangeComponent(comp, true);
        pendingBuffers_.push_back(comp);
    }
}

void HaloExchanger::waitExchange() {
    if (requests_.empty()) return;

    // Wait for all pending requests
    std::vector<MPI_Status> statuses(requests_.size());
    MPI_Waitall(static_cast<int>(requests_.size()), requests_.data(), statuses.data());

    // If using host staging, copy received data back to GPU
    if (!cudaAwareMPI_) {
        size_t planeBytes = grid_.boundaryPlaneBytes();

        for (real* data : pendingBuffers_) {
            // Copy lower halo from host to device
            if (grid_.hasLowerNeighbor()) {
                real* lowerHalo = data + grid_.lowerHaloStart() * grid_.planeSize();
                cudaMemcpy(lowerHalo, recvLowerHost_.data(), planeBytes,
                           cudaMemcpyHostToDevice);
            }

            // Copy upper halo from host to device
            if (grid_.hasUpperNeighbor()) {
                real* upperHalo = data + grid_.upperHaloStart() * grid_.planeSize();
                cudaMemcpy(upperHalo, recvUpperHost_.data(), planeBytes,
                           cudaMemcpyHostToDevice);
            }
        }
    }

    requests_.clear();
    pendingBuffers_.clear();
}

// ============================================================================
// Internal exchange implementation
// ============================================================================

void HaloExchanger::exchangeComponent(real* data, bool async) {
    int rank = MPIContext::rank();
    int lowerNeighbor = MPIContext::lowerNeighbor();
    int upperNeighbor = MPIContext::upperNeighbor();
    MPI_Comm comm = MPIContext::comm();

    size_t planeElements = grid_.planeSize() * grid_.haloWidth();
    size_t planeBytes = planeElements * sizeof(real);

    // Compute buffer offsets
    int realStart = grid_.realDataStart();
    int realEnd = grid_.upperHaloStart();

    // Pointers to boundary data (what we send)
    real* lowerBoundary = data + realStart * grid_.planeSize();
    real* upperBoundary = data + (realEnd - grid_.haloWidth()) * grid_.planeSize();

    // Pointers to halo regions (where we receive)
    real* lowerHalo = data + grid_.lowerHaloStart() * grid_.planeSize();
    real* upperHalo = data + grid_.upperHaloStart() * grid_.planeSize();

    if (cudaAwareMPI_) {
        // Direct GPU-to-GPU transfers via CUDA-aware MPI
        cudaDeviceSynchronize();  // Ensure GPU operations complete

        if (async) {
            MPI_Request req;

            // Send lower boundary to lower neighbor's upper halo
            if (lowerNeighbor >= 0) {
                MPI_Isend(lowerBoundary, planeElements, MPI_FLOAT,
                          lowerNeighbor, TAG_UPPER_TO_LOWER, comm, &req);
                requests_.push_back(req);

                MPI_Irecv(lowerHalo, planeElements, MPI_FLOAT,
                          lowerNeighbor, TAG_LOWER_TO_UPPER, comm, &req);
                requests_.push_back(req);
            }

            // Send upper boundary to upper neighbor's lower halo
            if (upperNeighbor >= 0) {
                MPI_Isend(upperBoundary, planeElements, MPI_FLOAT,
                          upperNeighbor, TAG_LOWER_TO_UPPER, comm, &req);
                requests_.push_back(req);

                MPI_Irecv(upperHalo, planeElements, MPI_FLOAT,
                          upperNeighbor, TAG_UPPER_TO_LOWER, comm, &req);
                requests_.push_back(req);
            }
        } else {
            // Blocking exchange using Sendrecv
            if (lowerNeighbor >= 0) {
                MPI_Sendrecv(lowerBoundary, planeElements, MPI_FLOAT,
                             lowerNeighbor, TAG_UPPER_TO_LOWER,
                             lowerHalo, planeElements, MPI_FLOAT,
                             lowerNeighbor, TAG_LOWER_TO_UPPER,
                             comm, MPI_STATUS_IGNORE);
            }

            if (upperNeighbor >= 0) {
                MPI_Sendrecv(upperBoundary, planeElements, MPI_FLOAT,
                             upperNeighbor, TAG_LOWER_TO_UPPER,
                             upperHalo, planeElements, MPI_FLOAT,
                             upperNeighbor, TAG_UPPER_TO_LOWER,
                             comm, MPI_STATUS_IGNORE);
            }
        }
    } else {
        // Host-staged transfers
        // Copy boundaries to host
        if (lowerNeighbor >= 0) {
            cudaMemcpy(sendLowerHost_.data(), lowerBoundary, planeBytes,
                       cudaMemcpyDeviceToHost);
        }
        if (upperNeighbor >= 0) {
            cudaMemcpy(sendUpperHost_.data(), upperBoundary, planeBytes,
                       cudaMemcpyDeviceToHost);
        }

        if (async) {
            MPI_Request req;

            if (lowerNeighbor >= 0) {
                MPI_Isend(sendLowerHost_.data(), planeElements, MPI_FLOAT,
                          lowerNeighbor, TAG_UPPER_TO_LOWER, comm, &req);
                requests_.push_back(req);

                MPI_Irecv(recvLowerHost_.data(), planeElements, MPI_FLOAT,
                          lowerNeighbor, TAG_LOWER_TO_UPPER, comm, &req);
                requests_.push_back(req);
            }

            if (upperNeighbor >= 0) {
                MPI_Isend(sendUpperHost_.data(), planeElements, MPI_FLOAT,
                          upperNeighbor, TAG_LOWER_TO_UPPER, comm, &req);
                requests_.push_back(req);

                MPI_Irecv(recvUpperHost_.data(), planeElements, MPI_FLOAT,
                          upperNeighbor, TAG_UPPER_TO_LOWER, comm, &req);
                requests_.push_back(req);
            }
            // Note: waitExchange() will copy received data back to GPU
        } else {
            // Blocking exchange
            if (lowerNeighbor >= 0) {
                MPI_Sendrecv(sendLowerHost_.data(), planeElements, MPI_FLOAT,
                             lowerNeighbor, TAG_UPPER_TO_LOWER,
                             recvLowerHost_.data(), planeElements, MPI_FLOAT,
                             lowerNeighbor, TAG_LOWER_TO_UPPER,
                             comm, MPI_STATUS_IGNORE);

                cudaMemcpy(lowerHalo, recvLowerHost_.data(), planeBytes,
                           cudaMemcpyHostToDevice);
            }

            if (upperNeighbor >= 0) {
                MPI_Sendrecv(sendUpperHost_.data(), planeElements, MPI_FLOAT,
                             upperNeighbor, TAG_LOWER_TO_UPPER,
                             recvUpperHost_.data(), planeElements, MPI_FLOAT,
                             upperNeighbor, TAG_UPPER_TO_LOWER,
                             comm, MPI_STATUS_IGNORE);

                cudaMemcpy(upperHalo, recvUpperHost_.data(), planeBytes,
                           cudaMemcpyHostToDevice);
            }
        }
    }
}

}  // namespace mumaxplus
