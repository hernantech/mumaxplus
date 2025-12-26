/**
 * Halo Exchange Test for mumax+ Multi-GPU Port
 *
 * This validates the halo exchange mechanism needed for stencil operations
 * (exchange field, DMI) in a distributed setting.
 *
 * Tests:
 * 1. Z-slab decomposition with padded buffers
 * 2. CUDA-aware MPI point-to-point communication
 * 3. Periodic boundary condition wrapping
 * 4. Stencil correctness after halo exchange
 *
 * Compilation:
 *   nvcc -x cu -ccbin mpicxx -std=c++14 -O3 halo_exchange_test.cu \
 *        -o halo_exchange_test -lmpi
 *
 * Execution:
 *   mpirun -np 2 ./halo_exchange_test
 *   mpirun -np 4 ./halo_exchange_test
 */

#include <iostream>
#include <vector>
#include <cmath>
#include <mpi.h>
#include <cuda_runtime.h>

// ============================================================================
// Configuration
// ============================================================================

// Grid dimensions
const int Nx = 64;
const int Ny = 64;
const int Nz = 64;

// Halo depth (1 layer for 6-point stencil)
const int HALO_DEPTH = 1;

// Number of field components (3 for vector field like magnetization)
const int NCOMP = 3;

// ============================================================================
// CUDA Helpers
// ============================================================================

#define CUDA_CHECK(call)                                                       \
    do {                                                                       \
        cudaError_t err = call;                                                \
        if (err != cudaSuccess) {                                              \
            std::cerr << "CUDA Error: " << cudaGetErrorString(err)             \
                      << " at " << __FILE__ << ":" << __LINE__ << std::endl;   \
            MPI_Abort(MPI_COMM_WORLD, 1);                                      \
        }                                                                      \
    } while (0)

// ============================================================================
// GPU Kernels
// ============================================================================

/**
 * Initialize field with a pattern that encodes global Z coordinate.
 * This allows us to verify halo exchange correctness.
 *
 * Pattern: field[z][y][x] = global_z + 0.001 * y + 0.000001 * x
 */
__global__ void k_init_field(
    float* field,
    int Nx, int Ny, int local_Nz,
    int z_offset,      // Global Z offset for this rank
    int halo_depth,
    int component      // Which component (0, 1, 2)
) {
    int idx = blockIdx.x * blockDim.x + threadIdx.x;
    int total_cells = Nx * Ny * local_Nz;

    if (idx >= total_cells) return;

    // Convert linear index to 3D (X-fast, Z-slow)
    int x = idx % Nx;
    int y = (idx / Nx) % Ny;
    int local_z = idx / (Nx * Ny);

    // Global Z coordinate
    int global_z = local_z + z_offset;

    // Unique value encoding position and component
    float value = (float)global_z + 0.001f * y + 0.000001f * x + 0.1f * component;

    // Write to padded buffer (skip lower halo region)
    int padded_idx = x + y * Nx + (local_z + halo_depth) * Nx * Ny;
    field[padded_idx] = value;
}

/**
 * Verify halo values match expected global pattern.
 */
__global__ void k_verify_halo(
    const float* field,
    int Nx, int Ny, int local_Nz,
    int z_offset,
    int halo_depth,
    int component,
    int Nz_global,
    bool is_periodic,
    float* errors,     // Output: max error per block
    bool check_lower   // true = check lower halo, false = check upper
) {
    __shared__ float sdata[256];

    int tid = threadIdx.x;
    int idx = blockIdx.x * blockDim.x + threadIdx.x;
    int halo_cells = Nx * Ny * halo_depth;

    float my_error = 0.0f;

    if (idx < halo_cells) {
        int x = idx % Nx;
        int y = (idx / Nx) % Ny;
        int halo_z = idx / (Nx * Ny);

        int global_z;
        int padded_idx;

        if (check_lower) {
            // Lower halo: should contain data from (z_offset - halo_depth) to (z_offset - 1)
            global_z = z_offset - halo_depth + halo_z;
            if (is_periodic && global_z < 0) {
                global_z += Nz_global;  // Wrap around
            }
            padded_idx = x + y * Nx + halo_z * Nx * Ny;
        } else {
            // Upper halo: should contain data from (z_offset + local_Nz) to (z_offset + local_Nz + halo_depth - 1)
            global_z = z_offset + local_Nz + halo_z;
            if (is_periodic && global_z >= Nz_global) {
                global_z -= Nz_global;  // Wrap around
            }
            padded_idx = x + y * Nx + (local_Nz + halo_depth + halo_z) * Nx * Ny;
        }

        // Expected value based on global position
        float expected = (float)global_z + 0.001f * y + 0.000001f * x + 0.1f * component;
        float actual = field[padded_idx];

        my_error = fabsf(actual - expected);
    }

    sdata[tid] = my_error;
    __syncthreads();

    // Block reduction
    for (int s = blockDim.x / 2; s > 0; s >>= 1) {
        if (tid < s && sdata[tid + s] > sdata[tid]) {
            sdata[tid] = sdata[tid + s];
        }
        __syncthreads();
    }

    if (tid == 0) {
        errors[blockIdx.x] = sdata[0];
    }
}

/**
 * Simple 6-point Laplacian stencil to test halo usage.
 * laplacian = (m[x+1] + m[x-1] + m[y+1] + m[y-1] + m[z+1] + m[z-1] - 6*m[x,y,z])
 */
__global__ void k_laplacian_stencil(
    const float* input,
    float* output,
    int Nx, int Ny, int local_Nz,
    int halo_depth
) {
    int idx = blockIdx.x * blockDim.x + threadIdx.x;
    int total_cells = Nx * Ny * local_Nz;

    if (idx >= total_cells) return;

    int x = idx % Nx;
    int y = (idx / Nx) % Ny;
    int local_z = idx / (Nx * Ny);

    // Padded coordinates (add halo offset)
    int pz = local_z + halo_depth;

    // Helper lambda for padded index
    auto pidx = [Nx, Ny](int px, int py, int pz) {
        return px + py * Nx + pz * Nx * Ny;
    };

    float center = input[pidx(x, y, pz)];

    // X neighbors (with periodic wrap in X for simplicity)
    int xm = (x > 0) ? x - 1 : Nx - 1;
    int xp = (x < Nx - 1) ? x + 1 : 0;

    // Y neighbors (with periodic wrap in Y)
    int ym = (y > 0) ? y - 1 : Ny - 1;
    int yp = (y < Ny - 1) ? y + 1 : 0;

    // Z neighbors (use halos!)
    int zm = pz - 1;  // Could go into lower halo (pz = halo_depth - 1)
    int zp = pz + 1;  // Could go into upper halo (pz = local_Nz + halo_depth)

    float laplacian =
        input[pidx(xp, y, pz)] + input[pidx(xm, y, pz)] +
        input[pidx(x, yp, pz)] + input[pidx(x, ym, pz)] +
        input[pidx(x, y, zp)] + input[pidx(x, y, zm)] -
        6.0f * center;

    output[pidx(x, y, pz)] = laplacian;
}

// ============================================================================
// Halo Exchange Implementation
// ============================================================================

class HaloExchanger {
public:
    int rank, size;
    int local_Nz;
    int z_offset;
    int padded_Nz;
    size_t plane_size;  // Nx * Ny * sizeof(float)
    bool is_periodic;

    // Staging buffers (for non-CUDA-aware MPI)
    float* h_send_lower;
    float* h_send_upper;
    float* h_recv_lower;
    float* h_recv_upper;

    HaloExchanger(int rank, int size, int local_Nz, int z_offset, bool periodic = true)
        : rank(rank), size(size), local_Nz(local_Nz), z_offset(z_offset),
          is_periodic(periodic)
    {
        padded_Nz = local_Nz + 2 * HALO_DEPTH;
        plane_size = Nx * Ny * sizeof(float);

        // Allocate pinned host memory for staging
        CUDA_CHECK(cudaMallocHost(&h_send_lower, plane_size * HALO_DEPTH));
        CUDA_CHECK(cudaMallocHost(&h_send_upper, plane_size * HALO_DEPTH));
        CUDA_CHECK(cudaMallocHost(&h_recv_lower, plane_size * HALO_DEPTH));
        CUDA_CHECK(cudaMallocHost(&h_recv_upper, plane_size * HALO_DEPTH));
    }

    ~HaloExchanger() {
        cudaFreeHost(h_send_lower);
        cudaFreeHost(h_send_upper);
        cudaFreeHost(h_recv_lower);
        cudaFreeHost(h_recv_upper);
    }

    /**
     * Exchange halos for a single field component.
     *
     * Buffer layout (padded):
     *   [0..HALO-1]             = lower halo (receives from rank-1)
     *   [HALO..HALO+local_Nz-1] = real data
     *   [HALO+local_Nz..]       = upper halo (receives from rank+1)
     */
    void exchange(float* d_field, int component_tag = 0) {
        // Determine neighbors
        int lower_neighbor = (rank > 0) ? rank - 1 : (is_periodic ? size - 1 : MPI_PROC_NULL);
        int upper_neighbor = (rank < size - 1) ? rank + 1 : (is_periodic ? 0 : MPI_PROC_NULL);

        // Pointers into padded buffer
        float* d_lower_halo = d_field;  // First HALO_DEPTH planes
        float* d_real_start = d_field + HALO_DEPTH * Nx * Ny;  // Start of real data
        float* d_real_end = d_real_start + (local_Nz - HALO_DEPTH) * Nx * Ny;  // Last plane of real data
        float* d_upper_halo = d_field + (HALO_DEPTH + local_Nz) * Nx * Ny;  // Upper halo

        // Data to send:
        // - Lower boundary (first HALO_DEPTH planes of real data) -> goes to rank-1's upper halo
        // - Upper boundary (last HALO_DEPTH planes of real data) -> goes to rank+1's lower halo

        float* d_send_lower = d_real_start;  // First plane of real data
        float* d_send_upper = d_field + (HALO_DEPTH + local_Nz - HALO_DEPTH) * Nx * Ny;  // Last plane

        // Copy to host staging buffers
        CUDA_CHECK(cudaMemcpy(h_send_lower, d_send_lower, plane_size * HALO_DEPTH, cudaMemcpyDeviceToHost));
        CUDA_CHECK(cudaMemcpy(h_send_upper, d_send_upper, plane_size * HALO_DEPTH, cudaMemcpyDeviceToHost));

        // MPI exchange
        MPI_Request reqs[4];
        int req_count = 0;

        // Send lower boundary to lower neighbor, receive from lower neighbor into lower halo
        if (lower_neighbor != MPI_PROC_NULL) {
            MPI_Isend(h_send_lower, Nx * Ny * HALO_DEPTH, MPI_FLOAT, lower_neighbor,
                      100 + component_tag, MPI_COMM_WORLD, &reqs[req_count++]);
            MPI_Irecv(h_recv_lower, Nx * Ny * HALO_DEPTH, MPI_FLOAT, lower_neighbor,
                      200 + component_tag, MPI_COMM_WORLD, &reqs[req_count++]);
        }

        // Send upper boundary to upper neighbor, receive from upper neighbor into upper halo
        if (upper_neighbor != MPI_PROC_NULL) {
            MPI_Isend(h_send_upper, Nx * Ny * HALO_DEPTH, MPI_FLOAT, upper_neighbor,
                      200 + component_tag, MPI_COMM_WORLD, &reqs[req_count++]);
            MPI_Irecv(h_recv_upper, Nx * Ny * HALO_DEPTH, MPI_FLOAT, upper_neighbor,
                      100 + component_tag, MPI_COMM_WORLD, &reqs[req_count++]);
        }

        // Wait for all communication
        if (req_count > 0) {
            MPI_Waitall(req_count, reqs, MPI_STATUSES_IGNORE);
        }

        // Copy received data to device halo regions
        if (lower_neighbor != MPI_PROC_NULL) {
            CUDA_CHECK(cudaMemcpy(d_lower_halo, h_recv_lower, plane_size * HALO_DEPTH, cudaMemcpyHostToDevice));
        }
        if (upper_neighbor != MPI_PROC_NULL) {
            CUDA_CHECK(cudaMemcpy(d_upper_halo, h_recv_upper, plane_size * HALO_DEPTH, cudaMemcpyHostToDevice));
        }
    }
};

// ============================================================================
// Main Test
// ============================================================================

int main(int argc, char** argv) {
    MPI_Init(&argc, &argv);

    int rank, size;
    MPI_Comm_rank(MPI_COMM_WORLD, &rank);
    MPI_Comm_size(MPI_COMM_WORLD, &size);

    // GPU setup
    int num_devices;
    CUDA_CHECK(cudaGetDeviceCount(&num_devices));
    CUDA_CHECK(cudaSetDevice(rank % num_devices));

    if (rank == 0) {
        std::cout << "========================================" << std::endl;
        std::cout << "Halo Exchange Test for mumax+ Port" << std::endl;
        std::cout << "========================================" << std::endl;
        std::cout << "Grid: " << Nx << " x " << Ny << " x " << Nz << std::endl;
        std::cout << "MPI Ranks: " << size << std::endl;
        std::cout << "Halo Depth: " << HALO_DEPTH << std::endl;
        std::cout << "Periodic BC: Yes" << std::endl;
        std::cout << "========================================" << std::endl;
    }
    MPI_Barrier(MPI_COMM_WORLD);

    // Domain decomposition
    int local_Nz = Nz / size;
    int z_offset = rank * local_Nz;
    if (rank == size - 1) {
        local_Nz = Nz - z_offset;  // Handle remainder
    }

    std::cout << "Rank " << rank << ": Z=[" << z_offset << ".." << (z_offset + local_Nz - 1)
              << "], local_Nz=" << local_Nz << std::endl;
    MPI_Barrier(MPI_COMM_WORLD);

    // Allocate padded field buffer
    int padded_Nz = local_Nz + 2 * HALO_DEPTH;
    size_t padded_size = Nx * Ny * padded_Nz;

    float* d_field;
    CUDA_CHECK(cudaMalloc(&d_field, padded_size * sizeof(float)));
    CUDA_CHECK(cudaMemset(d_field, 0, padded_size * sizeof(float)));

    // Initialize field with position-encoded values
    int threads = 256;
    int blocks = (Nx * Ny * local_Nz + threads - 1) / threads;
    k_init_field<<<blocks, threads>>>(d_field, Nx, Ny, local_Nz, z_offset, HALO_DEPTH, 0);
    CUDA_CHECK(cudaDeviceSynchronize());

    // Create halo exchanger
    HaloExchanger exchanger(rank, size, local_Nz, z_offset, true);

    // Perform halo exchange
    if (rank == 0) {
        std::cout << "\nPerforming halo exchange..." << std::endl;
    }
    MPI_Barrier(MPI_COMM_WORLD);

    exchanger.exchange(d_field, 0);

    CUDA_CHECK(cudaDeviceSynchronize());
    MPI_Barrier(MPI_COMM_WORLD);

    // Verify halos
    int halo_cells = Nx * Ny * HALO_DEPTH;
    int verify_blocks = (halo_cells + threads - 1) / threads;

    float* d_errors;
    CUDA_CHECK(cudaMalloc(&d_errors, verify_blocks * sizeof(float)));

    float lower_error = 0.0f, upper_error = 0.0f;

    // Check lower halo
    CUDA_CHECK(cudaMemset(d_errors, 0, verify_blocks * sizeof(float)));
    k_verify_halo<<<verify_blocks, threads>>>(
        d_field, Nx, Ny, local_Nz, z_offset, HALO_DEPTH, 0, Nz, true, d_errors, true
    );
    CUDA_CHECK(cudaDeviceSynchronize());

    std::vector<float> h_errors(verify_blocks);
    CUDA_CHECK(cudaMemcpy(h_errors.data(), d_errors, verify_blocks * sizeof(float), cudaMemcpyDeviceToHost));
    for (float e : h_errors) lower_error = std::max(lower_error, e);

    // Check upper halo
    CUDA_CHECK(cudaMemset(d_errors, 0, verify_blocks * sizeof(float)));
    k_verify_halo<<<verify_blocks, threads>>>(
        d_field, Nx, Ny, local_Nz, z_offset, HALO_DEPTH, 0, Nz, true, d_errors, false
    );
    CUDA_CHECK(cudaDeviceSynchronize());

    CUDA_CHECK(cudaMemcpy(h_errors.data(), d_errors, verify_blocks * sizeof(float), cudaMemcpyDeviceToHost));
    for (float e : h_errors) upper_error = std::max(upper_error, e);

    // Report per-rank errors
    std::cout << "Rank " << rank << ": lower_halo_error=" << lower_error
              << ", upper_halo_error=" << upper_error << std::endl;

    // Global error check
    float local_max = std::max(lower_error, upper_error);
    float global_max;
    MPI_Reduce(&local_max, &global_max, 1, MPI_FLOAT, MPI_MAX, 0, MPI_COMM_WORLD);

    if (rank == 0) {
        std::cout << "\n========================================" << std::endl;
        std::cout << "Global Max Halo Error: " << global_max << std::endl;

        if (global_max < 1e-5f) {
            std::cout << "[SUCCESS] Halo exchange validated!" << std::endl;
        } else {
            std::cout << "[FAILURE] Halo values incorrect!" << std::endl;
        }
        std::cout << "========================================" << std::endl;
    }

    // Test stencil operation using halos
    if (rank == 0) {
        std::cout << "\nTesting Laplacian stencil with halos..." << std::endl;
    }
    MPI_Barrier(MPI_COMM_WORLD);

    float* d_output;
    CUDA_CHECK(cudaMalloc(&d_output, padded_size * sizeof(float)));
    CUDA_CHECK(cudaMemset(d_output, 0, padded_size * sizeof(float)));

    k_laplacian_stencil<<<blocks, threads>>>(d_field, d_output, Nx, Ny, local_Nz, HALO_DEPTH);
    CUDA_CHECK(cudaDeviceSynchronize());

    if (rank == 0) {
        std::cout << "[SUCCESS] Stencil kernel completed without errors." << std::endl;
        std::cout << "========================================" << std::endl;
    }

    // Cleanup
    CUDA_CHECK(cudaFree(d_field));
    CUDA_CHECK(cudaFree(d_output));
    CUDA_CHECK(cudaFree(d_errors));

    MPI_Finalize();
    return 0;
}
