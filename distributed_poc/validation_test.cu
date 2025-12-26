/**
 * Multi-GPU vs Single-GPU Validation Test
 *
 * This test validates that the HeFFTe distributed FFT convolution produces
 * bit-identical (or near-identical) results compared to a single-GPU cuFFT
 * reference implementation.
 *
 * The test:
 * 1. Runs a single-GPU cuFFT convolution on rank 0 (reference)
 * 2. Runs a distributed HeFFTe convolution across all ranks
 * 3. Gathers the distributed result to rank 0
 * 4. Compares element-by-element, reporting max and RMS error
 *
 * This is critical for ensuring the mumax+ multi-GPU port produces
 * physically correct demagnetization field computations.
 *
 * Build:
 *   cmake -DHEFFTE_DIR=/path/to/heffte .. && make
 *
 * Run:
 *   mpirun -np 2 ./validation_test
 *   mpirun -np 4 ./validation_test
 *
 * Expected: Max error < 1e-5 (single precision tolerance)
 */

#include <iostream>
#include <vector>
#include <complex>
#include <cmath>
#include <cstdlib>
#include <numeric>
#include <mpi.h>
#include <cuda_runtime.h>
#include <cufft.h>
#include <heffte.h>

// ============================================================================
// Error Checking Macros
// ============================================================================

#define CUDA_CHECK(call)                                                       \
    do {                                                                       \
        cudaError_t err = call;                                                \
        if (err != cudaSuccess) {                                              \
            std::cerr << "CUDA Error at " << __FILE__ << ":" << __LINE__       \
                      << " - " << cudaGetErrorString(err) << std::endl;        \
            MPI_Abort(MPI_COMM_WORLD, 1);                                      \
        }                                                                      \
    } while (0)

#define CUFFT_CHECK(call)                                                      \
    do {                                                                       \
        cufftResult err = call;                                                \
        if (err != CUFFT_SUCCESS) {                                            \
            std::cerr << "cuFFT Error at " << __FILE__ << ":" << __LINE__      \
                      << " - Error code: " << err << std::endl;                \
            MPI_Abort(MPI_COMM_WORLD, 1);                                      \
        }                                                                      \
    } while (0)

// ============================================================================
// Test Parameters
// ============================================================================

// Grid size (64^3 is small enough to run quickly, large enough to be meaningful)
constexpr int NX = 64;
constexpr int NY = 64;
constexpr int NZ = 64;

// R2C output size in X dimension
constexpr int NX_COMPLEX = NX / 2 + 1;

// Total elements
constexpr long long TOTAL_REAL = (long long)NX * NY * NZ;
constexpr long long TOTAL_COMPLEX = (long long)NX_COMPLEX * NY * NZ;

// Tolerance for comparison (single precision)
constexpr float TOLERANCE = 1e-5f;

// ============================================================================
// GPU Kernels
// ============================================================================

/**
 * Initialize test data with a realistic Gaussian blob pattern.
 * This mimics a localized magnetization distribution.
 *
 * M(x,y,z) = exp(-((x-cx)^2 + (y-cy)^2 + (z-cz)^2) / (2*sigma^2))
 */
__global__ void k_init_gaussian(
    float* data,
    int nx, int ny, int nz,
    int z_offset,  // For distributed: starting z index
    int local_nz   // For distributed: number of local z planes
) {
    int idx = blockIdx.x * blockDim.x + threadIdx.x;
    int total_local = nx * ny * local_nz;

    if (idx >= total_local) return;

    // Compute 3D indices
    int x = idx % nx;
    int y = (idx / nx) % ny;
    int z = idx / (nx * ny) + z_offset;  // Global z coordinate

    // Gaussian parameters
    float cx = nx / 2.0f;
    float cy = ny / 2.0f;
    float cz = nz / 2.0f;
    float sigma = nx / 8.0f;

    float dx = x - cx;
    float dy = y - cy;
    float dz = z - cz;
    float r2 = dx*dx + dy*dy + dz*dz;

    data[idx] = expf(-r2 / (2.0f * sigma * sigma));
}

/**
 * Apply a dipole-like demagnetization kernel in frequency domain.
 *
 * For a dipolar kernel, K(k) ~ k_i * k_j / |k|^2 (for i,j components)
 * We use a simplified version: K(k) = 1 / (1 + |k|^2) to avoid singularity at k=0
 * This mimics the 1/r^3 decay of dipole fields in real space.
 */
__global__ void k_apply_dipole_kernel(
    float2* freq_data,
    int nx_complex, int ny, int nz,
    int z_offset,
    int local_nz,
    float normalization
) {
    int idx = blockIdx.x * blockDim.x + threadIdx.x;
    int total_local = nx_complex * ny * local_nz;

    if (idx >= total_local) return;

    // Compute 3D frequency indices
    int kx = idx % nx_complex;
    int ky = (idx / nx_complex) % ny;
    int kz_local = idx / (nx_complex * ny);
    int kz = kz_local + z_offset;  // Global kz coordinate

    // Compute wavenumbers (accounting for Nyquist frequencies)
    // For R2C: kx is already in [0, NX/2]
    // For ky, kz: wrap around for negative frequencies
    int nx_full = (nx_complex - 1) * 2;  // Original NX

    float k_x = (float)kx;
    float k_y = (ky <= ny/2) ? (float)ky : (float)(ky - ny);
    float k_z = (kz <= nz/2) ? (float)kz : (float)(kz - nz);

    // Normalized wavenumbers
    float kx_norm = k_x / nx_full;
    float ky_norm = k_y / ny;
    float kz_norm = k_z / nz;

    float k2 = kx_norm*kx_norm + ky_norm*ky_norm + kz_norm*kz_norm;

    // Dipole-like kernel: 1 / (1 + alpha * |k|^2)
    // alpha controls the decay rate
    float alpha = 100.0f;
    float kernel_val = 1.0f / (1.0f + alpha * k2);

    // Apply kernel and normalization
    float scale = kernel_val * normalization;
    freq_data[idx].x *= scale;
    freq_data[idx].y *= scale;
}

/**
 * Compute error statistics between two arrays
 */
__global__ void k_compute_errors(
    const float* result,
    const float* reference,
    float* abs_errors,
    float* sq_errors,
    int size
) {
    int idx = blockIdx.x * blockDim.x + threadIdx.x;
    if (idx < size) {
        float diff = result[idx] - reference[idx];
        abs_errors[idx] = fabsf(diff);
        sq_errors[idx] = diff * diff;
    }
}

/**
 * Find maximum value in array (parallel reduction)
 */
__global__ void k_reduce_max(
    const float* input,
    float* output,
    int size
) {
    extern __shared__ float sdata[];

    int tid = threadIdx.x;
    int idx = blockIdx.x * blockDim.x + threadIdx.x;

    // Load and find local max
    float my_max = 0.0f;
    for (int i = idx; i < size; i += blockDim.x * gridDim.x) {
        if (input[i] > my_max) my_max = input[i];
    }
    sdata[tid] = my_max;
    __syncthreads();

    // Reduction in shared memory
    for (int s = blockDim.x / 2; s > 0; s >>= 1) {
        if (tid < s && sdata[tid + s] > sdata[tid]) {
            sdata[tid] = sdata[tid + s];
        }
        __syncthreads();
    }

    if (tid == 0) {
        output[blockIdx.x] = sdata[0];
    }
}

/**
 * Sum array elements (parallel reduction)
 */
__global__ void k_reduce_sum(
    const float* input,
    float* output,
    int size
) {
    extern __shared__ float sdata[];

    int tid = threadIdx.x;
    int idx = blockIdx.x * blockDim.x + threadIdx.x;

    // Load and compute local sum
    float my_sum = 0.0f;
    for (int i = idx; i < size; i += blockDim.x * gridDim.x) {
        my_sum += input[i];
    }
    sdata[tid] = my_sum;
    __syncthreads();

    // Reduction in shared memory
    for (int s = blockDim.x / 2; s > 0; s >>= 1) {
        if (tid < s) {
            sdata[tid] += sdata[tid + s];
        }
        __syncthreads();
    }

    if (tid == 0) {
        output[blockIdx.x] = sdata[0];
    }
}

// ============================================================================
// Single-GPU Reference Implementation (cuFFT)
// ============================================================================

/**
 * Runs the complete convolution pipeline on a single GPU using cuFFT.
 * This serves as the ground-truth reference.
 *
 * @param d_result Output array (NX * NY * NZ floats, device memory)
 */
void single_gpu_reference(float* d_result) {
    // Allocate buffers
    float* d_input;
    cufftComplex* d_freq;

    size_t real_bytes = TOTAL_REAL * sizeof(float);
    size_t complex_bytes = TOTAL_COMPLEX * sizeof(cufftComplex);

    CUDA_CHECK(cudaMalloc(&d_input, real_bytes));
    CUDA_CHECK(cudaMalloc(&d_freq, complex_bytes));

    // Initialize with Gaussian blob
    int threads = 256;
    int blocks = (TOTAL_REAL + threads - 1) / threads;
    k_init_gaussian<<<blocks, threads>>>(d_input, NX, NY, NZ, 0, NZ);
    CUDA_CHECK(cudaDeviceSynchronize());

    // Create cuFFT plans
    cufftHandle plan_r2c, plan_c2r;

    // cuFFT uses row-major order: (nz, ny, nx) for 3D FFT with nx being fastest
    CUFFT_CHECK(cufftPlan3d(&plan_r2c, NZ, NY, NX, CUFFT_R2C));
    CUFFT_CHECK(cufftPlan3d(&plan_c2r, NZ, NY, NX, CUFFT_C2R));

    // Forward FFT: R2C
    CUFFT_CHECK(cufftExecR2C(plan_r2c, d_input, d_freq));
    CUDA_CHECK(cudaDeviceSynchronize());

    // Apply dipole kernel in frequency domain
    int complex_blocks = (TOTAL_COMPLEX + threads - 1) / threads;
    float normalization = 1.0f / (float)TOTAL_REAL;

    k_apply_dipole_kernel<<<complex_blocks, threads>>>(
        reinterpret_cast<float2*>(d_freq),
        NX_COMPLEX, NY, NZ,
        0,   // z_offset = 0 for single GPU
        NZ,  // local_nz = NZ for single GPU
        normalization
    );
    CUDA_CHECK(cudaDeviceSynchronize());

    // Backward FFT: C2R
    CUFFT_CHECK(cufftExecC2R(plan_c2r, d_freq, d_result));
    CUDA_CHECK(cudaDeviceSynchronize());

    // Cleanup
    CUFFT_CHECK(cufftDestroy(plan_r2c));
    CUFFT_CHECK(cufftDestroy(plan_c2r));
    CUDA_CHECK(cudaFree(d_input));
    CUDA_CHECK(cudaFree(d_freq));
}

// ============================================================================
// Multi-GPU Distributed Implementation (HeFFTe)
// ============================================================================

/**
 * Runs the distributed convolution using HeFFTe across all MPI ranks.
 *
 * @param d_local_result Local portion of result (device memory)
 * @param local_nz Number of Z planes owned by this rank
 * @param z_start Starting Z index for this rank
 * @param rank MPI rank
 * @param num_ranks Total MPI ranks
 */
void distributed_heffte(
    float* d_local_result,
    int local_nz,
    int z_start,
    int rank,
    int num_ranks
) {
    int z_end = z_start + local_nz - 1;

    // Define HeFFTe boxes
    heffte::box3d<> inbox  = {{0, 0, z_start}, {NX-1, NY-1, z_end}};
    heffte::box3d<> outbox = {{0, 0, z_start}, {NX_COMPLEX-1, NY-1, z_end}};

    const int r2c_direction = 0;  // X dimension for R2C reduction

    // Create HeFFTe plan
    heffte::fft3d_r2c<heffte::backend::cufft> fft(
        inbox, outbox, r2c_direction, MPI_COMM_WORLD
    );

    // Get buffer sizes
    size_t real_size = fft.size_inbox();
    size_t complex_size = fft.size_outbox();

    // Allocate buffers
    float* d_input;
    std::complex<float>* d_freq;

    CUDA_CHECK(cudaMalloc(&d_input, real_size * sizeof(float)));
    CUDA_CHECK(cudaMalloc(&d_freq, complex_size * sizeof(std::complex<float>)));

    // Initialize with same Gaussian blob pattern
    int threads = 256;
    int blocks = (real_size + threads - 1) / threads;
    k_init_gaussian<<<blocks, threads>>>(d_input, NX, NY, NZ, z_start, local_nz);
    CUDA_CHECK(cudaDeviceSynchronize());

    // Forward FFT
    fft.forward(d_input, d_freq);
    CUDA_CHECK(cudaDeviceSynchronize());

    // Apply dipole kernel (same as single-GPU)
    int complex_blocks = (complex_size + threads - 1) / threads;
    float normalization = 1.0f / (float)TOTAL_REAL;

    k_apply_dipole_kernel<<<complex_blocks, threads>>>(
        reinterpret_cast<float2*>(d_freq),
        NX_COMPLEX, NY, NZ,
        z_start,
        local_nz,
        normalization
    );
    CUDA_CHECK(cudaDeviceSynchronize());

    // Backward FFT
    fft.backward(d_freq, d_local_result);
    CUDA_CHECK(cudaDeviceSynchronize());

    // Cleanup
    CUDA_CHECK(cudaFree(d_input));
    CUDA_CHECK(cudaFree(d_freq));
}

// ============================================================================
// Main Program
// ============================================================================

int main(int argc, char** argv) {
    // Initialize MPI
    MPI_Init(&argc, &argv);

    int rank, num_ranks;
    MPI_Comm_rank(MPI_COMM_WORLD, &rank);
    MPI_Comm_size(MPI_COMM_WORLD, &num_ranks);

    // GPU device assignment
    int num_devices;
    CUDA_CHECK(cudaGetDeviceCount(&num_devices));

    if (num_devices == 0) {
        if (rank == 0) {
            std::cerr << "No CUDA devices found!" << std::endl;
        }
        MPI_Abort(MPI_COMM_WORLD, 1);
    }

    int my_device = rank % num_devices;
    CUDA_CHECK(cudaSetDevice(my_device));

    // Print header
    if (rank == 0) {
        std::cout << "========================================" << std::endl;
        std::cout << "Multi-GPU vs Single-GPU Validation Test" << std::endl;
        std::cout << "========================================" << std::endl;
        std::cout << "Grid: " << NX << " x " << NY << " x " << NZ << std::endl;
        std::cout << "MPI Ranks: " << num_ranks << std::endl;
        std::cout << "CUDA Devices: " << num_devices << std::endl;
        std::cout << std::endl;
    }
    MPI_Barrier(MPI_COMM_WORLD);

    // Domain decomposition (Z-slabs)
    int base_nz = NZ / num_ranks;
    int remainder = NZ % num_ranks;
    int local_nz = base_nz + (rank < remainder ? 1 : 0);
    int z_start = rank * base_nz + std::min(rank, remainder);

    long long local_cells = (long long)NX * NY * local_nz;

    if (rank == 0) {
        std::cout << "Domain decomposition:" << std::endl;
    }
    MPI_Barrier(MPI_COMM_WORLD);

    for (int r = 0; r < num_ranks; r++) {
        if (rank == r) {
            std::cout << "  Rank " << rank << ": Z=[" << z_start << ".."
                      << (z_start + local_nz - 1) << "], "
                      << local_cells << " cells, GPU " << my_device << std::endl;
        }
        MPI_Barrier(MPI_COMM_WORLD);
    }

    if (rank == 0) std::cout << std::endl;
    MPI_Barrier(MPI_COMM_WORLD);

    // ========================================================================
    // Step 1: Run single-GPU reference on rank 0
    // ========================================================================

    float* d_reference = nullptr;
    std::vector<float> h_reference;

    if (rank == 0) {
        std::cout << "Running single-GPU reference (cuFFT)..." << std::endl;

        CUDA_CHECK(cudaMalloc(&d_reference, TOTAL_REAL * sizeof(float)));

        auto t_start = MPI_Wtime();
        single_gpu_reference(d_reference);
        auto t_end = MPI_Wtime();

        std::cout << "  Time: " << (t_end - t_start) * 1000.0 << " ms" << std::endl;

        // Copy to host for later comparison
        h_reference.resize(TOTAL_REAL);
        CUDA_CHECK(cudaMemcpy(h_reference.data(), d_reference,
                              TOTAL_REAL * sizeof(float), cudaMemcpyDeviceToHost));
    }
    MPI_Barrier(MPI_COMM_WORLD);

    // ========================================================================
    // Step 2: Run distributed HeFFTe computation
    // ========================================================================

    if (rank == 0) {
        std::cout << "Running distributed computation (HeFFTe)..." << std::endl;
    }
    MPI_Barrier(MPI_COMM_WORLD);

    float* d_local_result;
    CUDA_CHECK(cudaMalloc(&d_local_result, local_cells * sizeof(float)));

    auto t_start = MPI_Wtime();
    distributed_heffte(d_local_result, local_nz, z_start, rank, num_ranks);
    auto t_end = MPI_Wtime();

    // Get timing from all ranks
    double local_time = t_end - t_start;
    double max_time;
    MPI_Reduce(&local_time, &max_time, 1, MPI_DOUBLE, MPI_MAX, 0, MPI_COMM_WORLD);

    if (rank == 0) {
        std::cout << "  Time: " << max_time * 1000.0 << " ms" << std::endl;
    }
    MPI_Barrier(MPI_COMM_WORLD);

    // ========================================================================
    // Step 3: Gather distributed result to rank 0
    // ========================================================================

    if (rank == 0) {
        std::cout << "Gathering distributed result..." << std::endl;
    }

    // Copy local result to host
    std::vector<float> h_local_result(local_cells);
    CUDA_CHECK(cudaMemcpy(h_local_result.data(), d_local_result,
                          local_cells * sizeof(float), cudaMemcpyDeviceToHost));

    // Gather all results to rank 0
    std::vector<float> h_distributed;
    if (rank == 0) {
        h_distributed.resize(TOTAL_REAL);
    }

    // Compute displacements and counts for MPI_Gatherv
    std::vector<int> recvcounts(num_ranks);
    std::vector<int> displs(num_ranks);

    int my_count = local_cells;
    MPI_Gather(&my_count, 1, MPI_INT, recvcounts.data(), 1, MPI_INT, 0, MPI_COMM_WORLD);

    if (rank == 0) {
        displs[0] = 0;
        for (int r = 1; r < num_ranks; r++) {
            displs[r] = displs[r-1] + recvcounts[r-1];
        }
    }

    // Gather all data
    MPI_Gatherv(h_local_result.data(), local_cells, MPI_FLOAT,
                h_distributed.data(), recvcounts.data(), displs.data(), MPI_FLOAT,
                0, MPI_COMM_WORLD);

    // ========================================================================
    // Step 4: Compare results on rank 0
    // ========================================================================

    if (rank == 0) {
        std::cout << std::endl;
        std::cout << "Comparison Results:" << std::endl;

        // Compute error statistics
        double max_abs_error = 0.0;
        double sum_sq_error = 0.0;
        double max_ref_val = 0.0;

        for (long long i = 0; i < TOTAL_REAL; i++) {
            double diff = std::abs((double)h_distributed[i] - (double)h_reference[i]);
            double sq_diff = diff * diff;

            if (diff > max_abs_error) max_abs_error = diff;
            sum_sq_error += sq_diff;

            if (std::abs(h_reference[i]) > max_ref_val) {
                max_ref_val = std::abs(h_reference[i]);
            }
        }

        double rms_error = std::sqrt(sum_sq_error / TOTAL_REAL);
        double relative_max_error = (max_ref_val > 0) ? max_abs_error / max_ref_val : max_abs_error;

        std::cout << "  Max Absolute Error: " << std::scientific << max_abs_error << std::endl;
        std::cout << "  RMS Error:          " << std::scientific << rms_error << std::endl;
        std::cout << "  Max Reference Value:" << std::scientific << max_ref_val << std::endl;
        std::cout << "  Relative Max Error: " << std::scientific << relative_max_error << std::endl;
        std::cout << std::endl;

        // Determine pass/fail
        bool pass = max_abs_error < TOLERANCE;

        if (pass) {
            std::cout << "[SUCCESS] Results match within tolerance ("
                      << TOLERANCE << ")" << std::endl;
        } else {
            std::cout << "[FAILURE] Results exceed tolerance!" << std::endl;
            std::cout << "  Expected max error < " << TOLERANCE << std::endl;
            std::cout << "  Actual max error:    " << max_abs_error << std::endl;

            // Find the first few locations with largest errors for debugging
            std::cout << std::endl;
            std::cout << "Sample differences (first 5 with error > tolerance/2):" << std::endl;
            int count = 0;
            for (long long i = 0; i < TOTAL_REAL && count < 5; i++) {
                double diff = std::abs((double)h_distributed[i] - (double)h_reference[i]);
                if (diff > TOLERANCE / 2) {
                    int x = i % NX;
                    int y = (i / NX) % NY;
                    int z = i / (NX * NY);
                    std::cout << "  [" << x << "," << y << "," << z << "]: "
                              << "ref=" << h_reference[i]
                              << ", dist=" << h_distributed[i]
                              << ", diff=" << diff << std::endl;
                    count++;
                }
            }
        }

        std::cout << "========================================" << std::endl;
    }

    // ========================================================================
    // Cleanup
    // ========================================================================

    CUDA_CHECK(cudaFree(d_local_result));
    if (rank == 0 && d_reference != nullptr) {
        CUDA_CHECK(cudaFree(d_reference));
    }

    MPI_Finalize();
    return 0;
}
