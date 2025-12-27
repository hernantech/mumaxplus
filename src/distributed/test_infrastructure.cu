/**
 * Infrastructure test for distributed multi-GPU module
 *
 * Tests:
 *   1. MPIContext initialization and decomposition
 *   2. DistributedGrid coordinate mapping
 *   3. HaloExchanger communication
 *
 * Build:
 *   mkdir build && cd build
 *   cmake -DENABLE_DISTRIBUTED=ON ..
 *   make test_infrastructure
 *
 * Run:
 *   mpirun -np 2 ./test_infrastructure
 */

#include "mpicontext.hpp"
#include "distributedgrid.hpp"
#include "haloexchanger.hpp"

#include <cuda_runtime.h>
#include <iostream>
#include <vector>
#include <cmath>
#include <cassert>

using namespace mumaxplus;

// Test grid dimensions
static constexpr int NX = 64;
static constexpr int NY = 64;
static constexpr int NZ = 32;

// Simple GPU kernel to fill a field with values based on global Z
__global__ void fillFieldKernel(real* data, int nx, int ny, int nz,
                                 int zStart, int haloOffset) {
    int x = blockIdx.x * blockDim.x + threadIdx.x;
    int y = blockIdx.y * blockDim.y + threadIdx.y;
    int z = blockIdx.z * blockDim.z + threadIdx.z;

    if (x < nx && y < ny && z < nz) {
        int localZ = z + haloOffset;
        int globalZ = zStart + z;
        int idx = x + y * nx + localZ * nx * ny;
        // Value encodes global Z position
        data[idx] = static_cast<real>(globalZ);
    }
}

// Check halo values
__global__ void checkHaloKernel(real* data, int nx, int ny,
                                 int haloStart, int haloWidth,
                                 int expectedZ, bool* success) {
    int x = blockIdx.x * blockDim.x + threadIdx.x;
    int y = blockIdx.y * blockDim.y + threadIdx.y;
    int z = blockIdx.z * blockDim.z + threadIdx.z;

    if (x < nx && y < ny && z < haloWidth) {
        int localZ = haloStart + z;
        int idx = x + y * nx + localZ * nx * ny;
        real expected = static_cast<real>(expectedZ + z);
        if (fabs(data[idx] - expected) > 1e-5) {
            *success = false;
        }
    }
}

bool testMPIContext(int argc, char** argv) {
    std::cout << "\n=== Test 1: MPIContext ===\n";

    // Should already be initialized
    if (!MPIContext::isInitialized()) {
        std::cerr << "FAIL: MPIContext not initialized\n";
        return false;
    }

    int rank = getMPIRank();
    int size = getMPISize();

    std::cout << "Rank " << rank << "/" << size
              << ": GPU " << MPIContext::deviceId() << std::endl;

    // Configure decomposition
    MPIContext::configureDecomposition(NZ);

    // Verify decomposition
    int totalNz = 0;
    int localNz = MPIContext::localNz();
    MPI_Allreduce(&localNz, &totalNz, 1, MPI_INT, MPI_SUM, MPIContext::comm());

    if (totalNz != NZ) {
        std::cerr << "FAIL: Total Nz mismatch: " << totalNz << " != " << NZ << "\n";
        return false;
    }

    // Verify non-overlapping ranges
    std::vector<int> allStarts(size);
    std::vector<int> allEnds(size);
    int zStart = MPIContext::zStart();
    int zEnd = MPIContext::zEnd();

    MPI_Allgather(&zStart, 1, MPI_INT, allStarts.data(), 1, MPI_INT, MPIContext::comm());
    MPI_Allgather(&zEnd, 1, MPI_INT, allEnds.data(), 1, MPI_INT, MPIContext::comm());

    if (rank == 0) {
        for (int r = 0; r < size - 1; r++) {
            if (allEnds[r] + 1 != allStarts[r + 1]) {
                std::cerr << "FAIL: Z ranges not contiguous\n";
                return false;
            }
        }
        if (allStarts[0] != 0 || allEnds[size - 1] != NZ - 1) {
            std::cerr << "FAIL: Z ranges don't cover full domain\n";
            return false;
        }
    }

    MPIContext::barrier();
    if (isMPIRoot()) std::cout << "PASS: MPIContext\n";
    return true;
}

bool testDistributedGrid() {
    std::cout << "\n=== Test 2: DistributedGrid ===\n";

    int rank = getMPIRank();

    int3 globalSize = {NX, NY, NZ};
    real3 cellSize = {1e-9f, 1e-9f, 1e-9f};
    int haloWidth = 1;

    DistributedGrid grid(globalSize, cellSize, haloWidth);

    // Verify global size
    if (grid.globalSize().x != NX || grid.globalSize().y != NY ||
        grid.globalSize().z != NZ) {
        std::cerr << "Rank " << rank << " FAIL: Global size mismatch\n";
        return false;
    }

    // Verify local size matches MPIContext
    if (grid.localNz() != MPIContext::localNz()) {
        std::cerr << "Rank " << rank << " FAIL: Local Nz mismatch\n";
        return false;
    }

    // Verify coordinate mapping
    for (int gz = grid.zStart(); gz <= grid.zEnd(); gz++) {
        int lz = grid.globalToLocalZ(gz);
        if (lz < 0) {
            std::cerr << "Rank " << rank << " FAIL: globalToLocalZ returned -1 for owned Z\n";
            return false;
        }
        int gz2 = grid.localToGlobalZ(lz);
        if (gz2 != gz) {
            std::cerr << "Rank " << rank << " FAIL: Round-trip Z mapping failed\n";
            return false;
        }
    }

    MPIContext::barrier();
    if (isMPIRoot()) std::cout << "PASS: DistributedGrid\n";
    return true;
}

bool testHaloExchanger() {
    std::cout << "\n=== Test 3: HaloExchanger ===\n";

    int rank = getMPIRank();
    int size = getMPISize();

    if (size < 2) {
        if (isMPIRoot()) {
            std::cout << "SKIP: HaloExchanger requires >= 2 ranks\n";
        }
        return true;
    }

    int3 globalSize = {NX, NY, NZ};
    real3 cellSize = {1e-9f, 1e-9f, 1e-9f};
    int haloWidth = 1;

    DistributedGrid grid(globalSize, cellSize, haloWidth);
    HaloExchanger exchanger(grid);

    // Allocate padded buffer on GPU
    size_t paddedElements = grid.paddedCellCount();
    real* d_data = nullptr;
    cudaMalloc(&d_data, paddedElements * sizeof(real));
    cudaMemset(d_data, 0, paddedElements * sizeof(real));

    // Fill real data region with global Z values
    dim3 blockDim(8, 8, 4);
    dim3 gridDim((NX + 7) / 8, (NY + 7) / 8, (grid.localNz() + 3) / 4);
    fillFieldKernel<<<gridDim, blockDim>>>(
        d_data, NX, NY, grid.localNz(),
        grid.zStart(), grid.realDataStart()
    );
    cudaDeviceSynchronize();

    // Perform halo exchange
    exchanger.exchange(d_data);
    cudaDeviceSynchronize();

    // Verify halos
    bool* d_success = nullptr;
    cudaMalloc(&d_success, sizeof(bool));
    bool h_success = true;
    cudaMemcpy(d_success, &h_success, sizeof(bool), cudaMemcpyHostToDevice);

    dim3 haloGridDim((NX + 7) / 8, (NY + 7) / 8, 1);

    // Check lower halo (should have upper boundary from lower neighbor)
    if (grid.hasLowerNeighbor()) {
        int expectedZ = grid.zStart() - haloWidth;
        checkHaloKernel<<<haloGridDim, blockDim>>>(
            d_data, NX, NY,
            grid.lowerHaloStart(), haloWidth,
            expectedZ, d_success
        );
    }

    // Check upper halo (should have lower boundary from upper neighbor)
    if (grid.hasUpperNeighbor()) {
        int expectedZ = grid.zEnd() + 1;
        checkHaloKernel<<<haloGridDim, blockDim>>>(
            d_data, NX, NY,
            grid.upperHaloStart(), haloWidth,
            expectedZ, d_success
        );
    }

    cudaDeviceSynchronize();
    cudaMemcpy(&h_success, d_success, sizeof(bool), cudaMemcpyDeviceToHost);

    cudaFree(d_data);
    cudaFree(d_success);

    // Gather results
    int localSuccess = h_success ? 1 : 0;
    int globalSuccess = 0;
    MPI_Allreduce(&localSuccess, &globalSuccess, 1, MPI_INT, MPI_MIN, MPIContext::comm());

    if (globalSuccess == 0) {
        std::cerr << "Rank " << rank << " FAIL: Halo values incorrect\n";
        return false;
    }

    MPIContext::barrier();
    if (isMPIRoot()) std::cout << "PASS: HaloExchanger\n";
    return true;
}

int main(int argc, char** argv) {
    // Initialize MPI and bind GPU
    MPIContext::initialize(argc, argv);

    if (isMPIRoot()) {
        std::cout << "===================================\n";
        std::cout << "Distributed Infrastructure Tests\n";
        std::cout << "Grid: " << NX << "x" << NY << "x" << NZ << "\n";
        std::cout << "Ranks: " << getMPISize() << "\n";
        std::cout << "===================================\n";
    }

    bool allPassed = true;

    allPassed &= testMPIContext(argc, argv);
    allPassed &= testDistributedGrid();
    allPassed &= testHaloExchanger();

    MPIContext::barrier();

    if (isMPIRoot()) {
        std::cout << "\n===================================\n";
        if (allPassed) {
            std::cout << "All tests PASSED\n";
        } else {
            std::cout << "Some tests FAILED\n";
        }
        std::cout << "===================================\n";
    }

    MPIContext::finalize();
    return allPassed ? 0 : 1;
}
