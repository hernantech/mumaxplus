/**
 * HeFFTe Distributed Micromagnetics Proof-of-Concept
 *
 * This validates the core distributed FFT convolution needed for
 * multi-GPU demagnetization field computation in mumax+.
 *
 * Build Requirements:
 * - HeFFTe (https://github.com/icl-utk-edu/heffte) with CUDA backend
 * - CUDA Toolkit 11.0+
 * - MPI (OpenMPI or MPICH with CUDA-aware support recommended)
 *
 * Compilation:
 *   nvcc -x cu -ccbin mpicxx -std=c++14 -O3 heffte_poc.cu -o heffte_poc \
 *        -I${HEFFTE_DIR}/include -L${HEFFTE_DIR}/lib -lheffte \
 *        -lcufft -lmpi
 *
 * Execution:
 *   mpirun -np 2 ./heffte_poc
 *   mpirun -np 4 ./heffte_poc
 *
 * Expected Output:
 *   - Max Error < 1e-5 indicates successful distributed convolution
 */

#include <iostream>
#include <vector>
#include <complex>
#include <cmath>
#include <cstdlib>
#include <mpi.h>
#include <cuda_runtime.h>
#include <heffte.h>

// ============================================================================
// CUDA Error Checking
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

// ============================================================================
// GPU Kernels
// ============================================================================

/**
 * Initialize test data with a known pattern.
 * Using constant 1.0 for easy validation of normalization.
 */
__global__ void k_init_data(float* data, int size) {
    int idx = blockIdx.x * blockDim.x + threadIdx.x;
    if (idx < size) {
        data[idx] = 1.0f;
    }
}

/**
 * Apply demagnetization kernel in frequency domain.
 *
 * In real mumax+, this would be:
 *   H_k[i] = K_xx[i]*M_x[i] + K_xy[i]*M_y[i] + K_xz[i]*M_z[i]  (for H_x)
 *   etc. for all 6 tensor components and 3 field components
 *
 * For this PoC, we use identity kernel with normalization:
 *   H = M * (1/N)
 *
 * This validates that the round-trip FFT->IFFT preserves data.
 */
__global__ void k_apply_demag_kernel(
    float* data_real,      // Real part (interleaved complex)
    float* data_imag,      // Imag part
    int size,
    float normalization    // 1.0 / total_cells
) {
    int idx = blockIdx.x * blockDim.x + threadIdx.x;
    if (idx < size) {
        // Identity kernel: just apply normalization
        // In real code: multiply by precomputed K_tensor[idx]
        data_real[idx] *= normalization;
        data_imag[idx] *= normalization;
    }
}

/**
 * Alternative kernel for std::complex<float> (interleaved real/imag)
 */
__global__ void k_apply_kernel_complex(
    float2* data,          // float2 = {real, imag}
    int size,
    float normalization
) {
    int idx = blockIdx.x * blockDim.x + threadIdx.x;
    if (idx < size) {
        data[idx].x *= normalization;
        data[idx].y *= normalization;
    }
}

/**
 * Compute max absolute error for validation
 */
__global__ void k_max_error(
    const float* data,
    float expected,
    float* block_max,
    int size
) {
    __shared__ float sdata[256];

    int tid = threadIdx.x;
    int idx = blockIdx.x * blockDim.x + threadIdx.x;

    // Each thread computes its max error
    float my_max = 0.0f;
    for (int i = idx; i < size; i += blockDim.x * gridDim.x) {
        float err = fabsf(data[i] - expected);
        if (err > my_max) my_max = err;
    }
    sdata[tid] = my_max;
    __syncthreads();

    // Reduction within block
    for (int s = blockDim.x / 2; s > 0; s >>= 1) {
        if (tid < s && sdata[tid + s] > sdata[tid]) {
            sdata[tid] = sdata[tid + s];
        }
        __syncthreads();
    }

    if (tid == 0) {
        block_max[blockIdx.x] = sdata[0];
    }
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

    // ========================================================================
    // 1. GPU Device Assignment
    // ========================================================================
    int num_devices;
    CUDA_CHECK(cudaGetDeviceCount(&num_devices));

    if (num_devices == 0) {
        if (rank == 0) {
            std::cerr << "No CUDA devices found!" << std::endl;
        }
        MPI_Abort(MPI_COMM_WORLD, 1);
    }

    // Assign GPUs round-robin to MPI ranks
    int my_device = rank % num_devices;
    CUDA_CHECK(cudaSetDevice(my_device));

    if (rank == 0) {
        std::cout << "========================================" << std::endl;
        std::cout << "HeFFTe Distributed FFT Proof-of-Concept" << std::endl;
        std::cout << "========================================" << std::endl;
        std::cout << "MPI Ranks: " << num_ranks << std::endl;
        std::cout << "CUDA Devices: " << num_devices << std::endl;
    }
    MPI_Barrier(MPI_COMM_WORLD);

    // Print device info per rank
    cudaDeviceProp prop;
    CUDA_CHECK(cudaGetDeviceProperties(&prop, my_device));
    std::cout << "Rank " << rank << " using GPU " << my_device
              << ": " << prop.name << std::endl;
    MPI_Barrier(MPI_COMM_WORLD);

    // ========================================================================
    // 2. Define Global Grid (Simulation Domain)
    // ========================================================================
    // For micromagnetics, typical sizes are 128³ to 1024³
    // Power-of-2 sizes are optimal for FFT performance

    const int Nx = 128;  // Fast dimension (contiguous in memory)
    const int Ny = 128;
    const int Nz = 128;  // Slow dimension (decomposed across GPUs)

    const long long total_cells = (long long)Nx * Ny * Nz;

    if (rank == 0) {
        std::cout << "\nGrid Size: " << Nx << " x " << Ny << " x " << Nz << std::endl;
        std::cout << "Total Cells: " << total_cells << std::endl;
    }

    // ========================================================================
    // 3. Domain Decomposition (Z-Slabs)
    // ========================================================================
    // Each rank owns a contiguous slab of Z-planes
    // This minimizes communication for row-major (X-fast) memory layout

    int base_nz = Nz / num_ranks;
    int remainder = Nz % num_ranks;

    // Distribute remainder to first 'remainder' ranks
    int local_Nz = base_nz + (rank < remainder ? 1 : 0);
    int z_start = rank * base_nz + std::min(rank, remainder);
    int z_end = z_start + local_Nz - 1;

    long long local_cells = (long long)Nx * Ny * local_Nz;

    std::cout << "Rank " << rank << ": Z=[" << z_start << ".." << z_end
              << "], local_Nz=" << local_Nz
              << ", cells=" << local_cells << std::endl;
    MPI_Barrier(MPI_COMM_WORLD);

    // Check for empty ranks (grid too small for num_ranks)
    if (local_Nz <= 0) {
        std::cerr << "Rank " << rank << " has no cells! Reduce MPI ranks." << std::endl;
        MPI_Abort(MPI_COMM_WORLD, 1);
    }

    // ========================================================================
    // 4. Define HeFFTe Boxes
    // ========================================================================
    // HeFFTe uses inclusive index ranges: box3d<>{{low}, {high}}
    // Coordinates are (x, y, z) order

    heffte::box3d<> inbox  = {{0, 0, z_start}, {Nx-1, Ny-1, z_end}};
    heffte::box3d<> outbox = inbox;  // Output same layout as input

    if (rank == 0) {
        std::cout << "\nCreating HeFFTe R2C plan..." << std::endl;
    }

    // ========================================================================
    // 5. Create HeFFTe FFT Plan
    // ========================================================================
    // R2C = Real-to-Complex (forward), C2R = Complex-to-Real (backward)
    // HeFFTe automatically determines optimal communication pattern

    heffte::fft3d_r2c<heffte::backend::cufft> fft(
        inbox,              // Input box (real space, my slab)
        outbox,             // Output box (real space, same layout)
        MPI_COMM_WORLD      // Communicator
    );

    // Get buffer sizes from HeFFTe
    size_t real_size = fft.size_inbox();      // Local real-space elements
    size_t complex_size = fft.size_outbox();  // Local freq-space elements

    if (rank == 0) {
        std::cout << "HeFFTe plan created." << std::endl;
        std::cout << "Real buffer size (per rank): " << real_size << std::endl;
        std::cout << "Complex buffer size (per rank): " << complex_size << std::endl;
    }

    // ========================================================================
    // 6. Allocate GPU Memory
    // ========================================================================
    float* d_input;
    float* d_output;
    std::complex<float>* d_freq;

    CUDA_CHECK(cudaMalloc(&d_input, real_size * sizeof(float)));
    CUDA_CHECK(cudaMalloc(&d_output, real_size * sizeof(float)));
    CUDA_CHECK(cudaMalloc(&d_freq, complex_size * sizeof(std::complex<float>)));

    // Initialize to zero
    CUDA_CHECK(cudaMemset(d_input, 0, real_size * sizeof(float)));
    CUDA_CHECK(cudaMemset(d_output, 0, real_size * sizeof(float)));
    CUDA_CHECK(cudaMemset(d_freq, 0, complex_size * sizeof(std::complex<float>)));

    // ========================================================================
    // 7. Initialize Test Data
    // ========================================================================
    // Set M = 1.0 everywhere for easy validation
    // After FFT -> identity kernel -> IFFT, result should be 1.0

    int threads = 256;
    int blocks = (real_size + threads - 1) / threads;
    k_init_data<<<blocks, threads>>>(d_input, real_size);
    CUDA_CHECK(cudaDeviceSynchronize());

    if (rank == 0) {
        std::cout << "\nRunning distributed FFT convolution..." << std::endl;
    }
    MPI_Barrier(MPI_COMM_WORLD);

    // ========================================================================
    // 8. Forward FFT: M(r) -> M(k)
    // ========================================================================
    // HeFFTe handles all MPI communication internally (AllToAll)

    auto t_start = MPI_Wtime();

    fft.forward(d_input, d_freq);

    auto t_forward = MPI_Wtime();

    // ========================================================================
    // 9. Apply Demag Kernel in Frequency Domain
    // ========================================================================
    // H(k) = K(k) * M(k)
    // For validation, use identity kernel with normalization

    float normalization = 1.0f / (float)total_cells;

    // Cast to float2 for easier kernel
    int freq_blocks = (complex_size + threads - 1) / threads;
    k_apply_kernel_complex<<<freq_blocks, threads>>>(
        reinterpret_cast<float2*>(d_freq),
        complex_size,
        normalization
    );
    CUDA_CHECK(cudaDeviceSynchronize());

    auto t_kernel = MPI_Wtime();

    // ========================================================================
    // 10. Backward FFT: H(k) -> H(r)
    // ========================================================================

    fft.backward(d_freq, d_output);

    auto t_backward = MPI_Wtime();

    CUDA_CHECK(cudaDeviceSynchronize());
    MPI_Barrier(MPI_COMM_WORLD);

    // ========================================================================
    // 11. Validation
    // ========================================================================
    // Expected result: 1.0 everywhere (input was 1.0, kernel was identity)

    // Compute max error on GPU
    int error_blocks = std::min(256, (int)((real_size + threads - 1) / threads));
    float* d_block_errors;
    CUDA_CHECK(cudaMalloc(&d_block_errors, error_blocks * sizeof(float)));

    k_max_error<<<error_blocks, threads>>>(d_output, 1.0f, d_block_errors, real_size);
    CUDA_CHECK(cudaDeviceSynchronize());

    // Copy block errors to host and find max
    std::vector<float> h_block_errors(error_blocks);
    CUDA_CHECK(cudaMemcpy(h_block_errors.data(), d_block_errors,
                          error_blocks * sizeof(float), cudaMemcpyDeviceToHost));

    float local_max_error = 0.0f;
    for (float e : h_block_errors) {
        if (e > local_max_error) local_max_error = e;
    }

    // Global reduction across all ranks
    float global_max_error;
    MPI_Reduce(&local_max_error, &global_max_error, 1, MPI_FLOAT, MPI_MAX, 0, MPI_COMM_WORLD);

    // ========================================================================
    // 12. Report Results
    // ========================================================================

    if (rank == 0) {
        std::cout << "\n========================================" << std::endl;
        std::cout << "Results" << std::endl;
        std::cout << "========================================" << std::endl;
        std::cout << "Forward FFT time:  " << (t_forward - t_start) * 1000 << " ms" << std::endl;
        std::cout << "Kernel apply time: " << (t_kernel - t_forward) * 1000 << " ms" << std::endl;
        std::cout << "Backward FFT time: " << (t_backward - t_kernel) * 1000 << " ms" << std::endl;
        std::cout << "Total time:        " << (t_backward - t_start) * 1000 << " ms" << std::endl;
        std::cout << "\nMax Error: " << global_max_error << std::endl;

        if (global_max_error < 1e-5f) {
            std::cout << "\n[SUCCESS] Distributed FFT convolution validated!" << std::endl;
            std::cout << "The round-trip FFT preserves data correctly." << std::endl;
        } else if (global_max_error < 1e-3f) {
            std::cout << "\n[WARNING] Error slightly high but acceptable." << std::endl;
            std::cout << "May be floating-point accumulation." << std::endl;
        } else {
            std::cout << "\n[FAILURE] Error too high!" << std::endl;
            std::cout << "Check FFT normalization and data layout." << std::endl;
        }
        std::cout << "========================================" << std::endl;
    }

    // ========================================================================
    // 13. Cleanup
    // ========================================================================

    CUDA_CHECK(cudaFree(d_input));
    CUDA_CHECK(cudaFree(d_output));
    CUDA_CHECK(cudaFree(d_freq));
    CUDA_CHECK(cudaFree(d_block_errors));

    MPI_Finalize();
    return 0;
}
