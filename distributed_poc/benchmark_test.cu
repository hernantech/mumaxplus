/**
 * Communication Overhead Benchmark for mumax+ Multi-GPU Port
 *
 * Measures:
 * 1. HeFFTe FFT transpose/communication overhead
 * 2. Halo exchange latency
 * 3. Compute vs communication ratio
 * 4. Scaling efficiency
 *
 * Usage:
 *   mpirun -np 2 ./benchmark_test [grid_size] [iterations]
 *   mpirun -np 4 ./benchmark_test 256 100
 */

#include <iostream>
#include <iomanip>
#include <vector>
#include <complex>
#include <cmath>
#include <algorithm>
#include <numeric>
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

__global__ void k_init_gaussian(float* data, int Nx, int Ny, int Nz,
                                 int z_start, float sigma) {
    int idx = blockIdx.x * blockDim.x + threadIdx.x;
    int total = Nx * Ny * Nz;
    if (idx >= total) return;

    int x = idx % Nx;
    int y = (idx / Nx) % Ny;
    int z = idx / (Nx * Ny) + z_start;

    float cx = Nx / 2.0f, cy = Ny / 2.0f, cz = (z_start + Nz/2);
    float r2 = (x - cx)*(x - cx) + (y - cy)*(y - cy) + (z - cz)*(z - cz);
    data[idx] = expf(-r2 / (2.0f * sigma * sigma));
}

__global__ void k_apply_kernel_complex(float2* data, int size, float norm) {
    int idx = blockIdx.x * blockDim.x + threadIdx.x;
    if (idx < size) {
        data[idx].x *= norm;
        data[idx].y *= norm;
    }
}

__global__ void k_laplacian_interior(const float* in, float* out,
                                      int Nx, int Ny, int local_Nz,
                                      int halo_size) {
    int idx = blockIdx.x * blockDim.x + threadIdx.x;
    int interior_size = Nx * Ny * local_Nz;
    if (idx >= interior_size) return;

    int x = idx % Nx;
    int y = (idx / Nx) % Ny;
    int z = idx / (Nx * Ny);

    // Map to padded coordinates (skip lower halo)
    int pz = z + halo_size;

    // Padded index
    int pidx = x + y * Nx + pz * (Nx * Ny);

    // 6-neighbor stencil with periodic BC in X/Y
    int xm = (x - 1 + Nx) % Nx;
    int xp = (x + 1) % Nx;
    int ym = (y - 1 + Ny) % Ny;
    int yp = (y + 1) % Ny;

    float center = in[pidx];
    float sum = in[xm + y * Nx + pz * Nx * Ny]
              + in[xp + y * Nx + pz * Nx * Ny]
              + in[x + ym * Nx + pz * Nx * Ny]
              + in[x + yp * Nx + pz * Nx * Ny]
              + in[x + y * Nx + (pz - 1) * Nx * Ny]
              + in[x + y * Nx + (pz + 1) * Nx * Ny];

    out[idx] = sum - 6.0f * center;
}

// ============================================================================
// Benchmark Results Structure
// ============================================================================

struct BenchmarkResults {
    // FFT timing (ms)
    double fft_forward_min, fft_forward_max, fft_forward_avg;
    double fft_backward_min, fft_backward_max, fft_backward_avg;
    double fft_kernel_avg;

    // Halo timing (ms)
    double halo_exchange_min, halo_exchange_max, halo_exchange_avg;

    // Compute timing (ms)
    double stencil_compute_avg;

    // Derived metrics
    double fft_comm_fraction;      // Communication / total FFT time
    double halo_comm_fraction;     // Halo exchange / stencil cycle time
    double effective_bandwidth_gb; // GB/s for halo exchange
};

// ============================================================================
// Halo Exchanger (simplified from halo_exchange_test.cu)
// ============================================================================

class HaloExchanger {
public:
    int rank, num_ranks;
    int Nx, Ny, local_Nz;
    int halo_size;
    size_t plane_size;  // Elements per Z-plane
    int padded_Nz;      // Total Z including halos

    float* h_send_lower;
    float* h_send_upper;
    float* h_recv_lower;
    float* h_recv_upper;

    HaloExchanger(int r, int nr, int nx, int ny, int lnz, int hs = 1)
        : rank(r), num_ranks(nr), Nx(nx), Ny(ny), local_Nz(lnz), halo_size(hs) {

        plane_size = (size_t)Nx * Ny * halo_size;
        padded_Nz = local_Nz + 2 * halo_size;

        CUDA_CHECK(cudaMallocHost(&h_send_lower, plane_size * sizeof(float)));
        CUDA_CHECK(cudaMallocHost(&h_send_upper, plane_size * sizeof(float)));
        CUDA_CHECK(cudaMallocHost(&h_recv_lower, plane_size * sizeof(float)));
        CUDA_CHECK(cudaMallocHost(&h_recv_upper, plane_size * sizeof(float)));
    }

    ~HaloExchanger() {
        cudaFreeHost(h_send_lower);
        cudaFreeHost(h_send_upper);
        cudaFreeHost(h_recv_lower);
        cudaFreeHost(h_recv_upper);
    }

    void exchange(float* d_field) {
        size_t plane_bytes = plane_size * sizeof(float);

        // Copy boundary planes to host
        // Lower boundary: first real plane (after halo)
        size_t lower_offset = halo_size * Nx * Ny;
        // Upper boundary: last real plane
        size_t upper_offset = (halo_size + local_Nz - 1) * Nx * Ny;

        CUDA_CHECK(cudaMemcpy(h_send_lower, d_field + lower_offset,
                              plane_bytes, cudaMemcpyDeviceToHost));
        CUDA_CHECK(cudaMemcpy(h_send_upper, d_field + upper_offset,
                              plane_bytes, cudaMemcpyDeviceToHost));

        // Determine neighbors (periodic)
        int lower_rank = (rank - 1 + num_ranks) % num_ranks;
        int upper_rank = (rank + 1) % num_ranks;

        MPI_Request reqs[4];
        int req_count = 0;

        // Non-blocking sends/recvs
        MPI_Irecv(h_recv_lower, plane_size, MPI_FLOAT, lower_rank, 1,
                  MPI_COMM_WORLD, &reqs[req_count++]);
        MPI_Irecv(h_recv_upper, plane_size, MPI_FLOAT, upper_rank, 0,
                  MPI_COMM_WORLD, &reqs[req_count++]);
        MPI_Isend(h_send_lower, plane_size, MPI_FLOAT, lower_rank, 0,
                  MPI_COMM_WORLD, &reqs[req_count++]);
        MPI_Isend(h_send_upper, plane_size, MPI_FLOAT, upper_rank, 1,
                  MPI_COMM_WORLD, &reqs[req_count++]);

        MPI_Waitall(req_count, reqs, MPI_STATUSES_IGNORE);

        // Copy received halos back to device
        // Lower halo: z = 0 to halo_size-1
        CUDA_CHECK(cudaMemcpy(d_field, h_recv_lower,
                              plane_bytes, cudaMemcpyHostToDevice));
        // Upper halo: z = halo_size + local_Nz to end
        size_t upper_halo_offset = (halo_size + local_Nz) * Nx * Ny;
        CUDA_CHECK(cudaMemcpy(d_field + upper_halo_offset, h_recv_upper,
                              plane_bytes, cudaMemcpyHostToDevice));
    }

    size_t bytes_exchanged() const {
        // Each rank sends 2 planes, receives 2 planes
        return 4 * plane_size * sizeof(float);
    }
};

// ============================================================================
// Main Benchmark
// ============================================================================

int main(int argc, char** argv) {
    MPI_Init(&argc, &argv);

    int rank, num_ranks;
    MPI_Comm_rank(MPI_COMM_WORLD, &rank);
    MPI_Comm_size(MPI_COMM_WORLD, &num_ranks);

    // Parse arguments
    int N = (argc > 1) ? atoi(argv[1]) : 128;
    int iterations = (argc > 2) ? atoi(argv[2]) : 50;
    int warmup = 5;

    // GPU setup
    int num_devices;
    CUDA_CHECK(cudaGetDeviceCount(&num_devices));
    int my_device = rank % num_devices;
    CUDA_CHECK(cudaSetDevice(my_device));

    if (rank == 0) {
        std::cout << "========================================" << std::endl;
        std::cout << "Communication Overhead Benchmark" << std::endl;
        std::cout << "========================================" << std::endl;
        std::cout << "Grid: " << N << " x " << N << " x " << N << std::endl;
        std::cout << "MPI Ranks: " << num_ranks << std::endl;
        std::cout << "GPUs per node: " << num_devices << std::endl;
        std::cout << "Iterations: " << iterations << " (warmup: " << warmup << ")" << std::endl;
        std::cout << std::endl;
    }

    // Domain decomposition
    int Nx = N, Ny = N, Nz = N;
    int base_nz = Nz / num_ranks;
    int remainder = Nz % num_ranks;
    int local_Nz = base_nz + (rank < remainder ? 1 : 0);
    int z_start = rank * base_nz + std::min(rank, remainder);

    long long total_cells = (long long)Nx * Ny * Nz;
    int halo_size = 1;
    int padded_Nz = local_Nz + 2 * halo_size;

    // ========================================================================
    // Setup HeFFTe
    // ========================================================================

    heffte::box3d<> inbox = {{0, 0, z_start}, {Nx-1, Ny-1, z_start + local_Nz - 1}};
    int out_Nx = Nx / 2 + 1;
    heffte::box3d<> outbox = {{0, 0, z_start}, {out_Nx-1, Ny-1, z_start + local_Nz - 1}};

    heffte::fft3d_r2c<heffte::backend::cufft> fft(inbox, outbox, 0, MPI_COMM_WORLD);

    size_t real_size = fft.size_inbox();
    size_t complex_size = fft.size_outbox();

    // Allocate FFT buffers
    float* d_fft_input;
    float* d_fft_output;
    std::complex<float>* d_freq;

    CUDA_CHECK(cudaMalloc(&d_fft_input, real_size * sizeof(float)));
    CUDA_CHECK(cudaMalloc(&d_fft_output, real_size * sizeof(float)));
    CUDA_CHECK(cudaMalloc(&d_freq, complex_size * sizeof(std::complex<float>)));

    // Initialize FFT input
    int threads = 256;
    int blocks = (real_size + threads - 1) / threads;
    k_init_gaussian<<<blocks, threads>>>(d_fft_input, Nx, Ny, local_Nz, z_start, N/8.0f);

    // ========================================================================
    // Setup Halo Exchange
    // ========================================================================

    HaloExchanger halo(rank, num_ranks, Nx, Ny, local_Nz, halo_size);

    // Allocate padded field for stencil
    size_t padded_size = (size_t)Nx * Ny * padded_Nz;
    float* d_padded_field;
    float* d_stencil_output;

    CUDA_CHECK(cudaMalloc(&d_padded_field, padded_size * sizeof(float)));
    CUDA_CHECK(cudaMalloc(&d_stencil_output, real_size * sizeof(float)));

    // Initialize padded field (copy real data, leave halos zero initially)
    CUDA_CHECK(cudaMemset(d_padded_field, 0, padded_size * sizeof(float)));
    CUDA_CHECK(cudaMemcpy(d_padded_field + halo_size * Nx * Ny,
                          d_fft_input, real_size * sizeof(float),
                          cudaMemcpyDeviceToDevice));

    // ========================================================================
    // Timing vectors
    // ========================================================================

    std::vector<double> fft_forward_times(iterations);
    std::vector<double> fft_backward_times(iterations);
    std::vector<double> fft_kernel_times(iterations);
    std::vector<double> halo_times(iterations);
    std::vector<double> stencil_times(iterations);

    float norm = 1.0f / (float)total_cells;
    int freq_blocks = (complex_size + threads - 1) / threads;
    int stencil_blocks = (real_size + threads - 1) / threads;

    // ========================================================================
    // Warmup
    // ========================================================================

    for (int i = 0; i < warmup; i++) {
        fft.forward(d_fft_input, d_freq);
        k_apply_kernel_complex<<<freq_blocks, threads>>>(
            reinterpret_cast<float2*>(d_freq), complex_size, norm);
        fft.backward(d_freq, d_fft_output);

        halo.exchange(d_padded_field);
        k_laplacian_interior<<<stencil_blocks, threads>>>(
            d_padded_field, d_stencil_output, Nx, Ny, local_Nz, halo_size);
    }
    CUDA_CHECK(cudaDeviceSynchronize());
    MPI_Barrier(MPI_COMM_WORLD);

    // ========================================================================
    // Benchmark Loop
    // ========================================================================

    for (int i = 0; i < iterations; i++) {
        double t0, t1, t2, t3, t4, t5;

        // --- FFT Forward ---
        MPI_Barrier(MPI_COMM_WORLD);
        t0 = MPI_Wtime();
        fft.forward(d_fft_input, d_freq);
        CUDA_CHECK(cudaDeviceSynchronize());
        MPI_Barrier(MPI_COMM_WORLD);
        t1 = MPI_Wtime();

        // --- FFT Kernel ---
        k_apply_kernel_complex<<<freq_blocks, threads>>>(
            reinterpret_cast<float2*>(d_freq), complex_size, norm);
        CUDA_CHECK(cudaDeviceSynchronize());
        t2 = MPI_Wtime();

        // --- FFT Backward ---
        MPI_Barrier(MPI_COMM_WORLD);
        fft.backward(d_freq, d_fft_output);
        CUDA_CHECK(cudaDeviceSynchronize());
        MPI_Barrier(MPI_COMM_WORLD);
        t3 = MPI_Wtime();

        // --- Halo Exchange ---
        MPI_Barrier(MPI_COMM_WORLD);
        t4 = MPI_Wtime();
        halo.exchange(d_padded_field);
        MPI_Barrier(MPI_COMM_WORLD);
        t4 = MPI_Wtime() - t4;

        // --- Stencil Compute ---
        t5 = MPI_Wtime();
        k_laplacian_interior<<<stencil_blocks, threads>>>(
            d_padded_field, d_stencil_output, Nx, Ny, local_Nz, halo_size);
        CUDA_CHECK(cudaDeviceSynchronize());
        t5 = MPI_Wtime() - t5;

        fft_forward_times[i] = (t1 - t0) * 1000.0;
        fft_kernel_times[i] = (t2 - t1) * 1000.0;
        fft_backward_times[i] = (t3 - t2) * 1000.0;
        halo_times[i] = t4 * 1000.0;
        stencil_times[i] = t5 * 1000.0;
    }

    // ========================================================================
    // Compute Statistics
    // ========================================================================

    // Compute statistics for each timing vector
    std::sort(fft_forward_times.begin(), fft_forward_times.end());
    std::sort(fft_backward_times.begin(), fft_backward_times.end());
    std::sort(fft_kernel_times.begin(), fft_kernel_times.end());
    std::sort(halo_times.begin(), halo_times.end());
    std::sort(stencil_times.begin(), stencil_times.end());

    double fwd_min = fft_forward_times.front();
    double fwd_max = fft_forward_times.back();
    double fwd_avg = std::accumulate(fft_forward_times.begin(), fft_forward_times.end(), 0.0) / iterations;

    double bwd_min = fft_backward_times.front();
    double bwd_max = fft_backward_times.back();
    double bwd_avg = std::accumulate(fft_backward_times.begin(), fft_backward_times.end(), 0.0) / iterations;

    double ker_avg = std::accumulate(fft_kernel_times.begin(), fft_kernel_times.end(), 0.0) / iterations;

    double halo_min = halo_times.front();
    double halo_max = halo_times.back();
    double halo_avg = std::accumulate(halo_times.begin(), halo_times.end(), 0.0) / iterations;

    double sten_avg = std::accumulate(stencil_times.begin(), stencil_times.end(), 0.0) / iterations;

    // Gather global stats
    double global_fwd_avg, global_bwd_avg, global_halo_avg, global_sten_avg;
    MPI_Reduce(&fwd_avg, &global_fwd_avg, 1, MPI_DOUBLE, MPI_MAX, 0, MPI_COMM_WORLD);
    MPI_Reduce(&bwd_avg, &global_bwd_avg, 1, MPI_DOUBLE, MPI_MAX, 0, MPI_COMM_WORLD);
    MPI_Reduce(&halo_avg, &global_halo_avg, 1, MPI_DOUBLE, MPI_MAX, 0, MPI_COMM_WORLD);
    MPI_Reduce(&sten_avg, &global_sten_avg, 1, MPI_DOUBLE, MPI_MAX, 0, MPI_COMM_WORLD);

    // ========================================================================
    // Report Results
    // ========================================================================

    if (rank == 0) {
        double total_fft_time = global_fwd_avg + ker_avg + global_bwd_avg;
        double fft_comm_time = global_fwd_avg + global_bwd_avg;  // Dominated by AllToAll

        double halo_bytes = halo.bytes_exchanged() * num_ranks;
        double halo_bw_gbps = (halo_bytes / 1e9) / (global_halo_avg / 1000.0);

        std::cout << "========================================" << std::endl;
        std::cout << "FFT Convolution Timing (ms)" << std::endl;
        std::cout << "========================================" << std::endl;
        std::cout << std::fixed << std::setprecision(3);
        std::cout << "Forward FFT:     " << std::setw(8) << global_fwd_avg
                  << " (min: " << fwd_min << ", max: " << fwd_max << ")" << std::endl;
        std::cout << "Kernel Apply:    " << std::setw(8) << ker_avg << std::endl;
        std::cout << "Backward FFT:    " << std::setw(8) << global_bwd_avg
                  << " (min: " << bwd_min << ", max: " << bwd_max << ")" << std::endl;
        std::cout << "Total FFT Cycle: " << std::setw(8) << total_fft_time << std::endl;
        std::cout << std::endl;

        std::cout << "========================================" << std::endl;
        std::cout << "Halo Exchange Timing (ms)" << std::endl;
        std::cout << "========================================" << std::endl;
        std::cout << "Halo Exchange:   " << std::setw(8) << global_halo_avg
                  << " (min: " << halo_min << ", max: " << halo_max << ")" << std::endl;
        std::cout << "Stencil Compute: " << std::setw(8) << global_sten_avg << std::endl;
        std::cout << "Data per rank:   " << std::setw(8)
                  << (halo.bytes_exchanged() / 1e6) << " MB" << std::endl;
        std::cout << "Effective BW:    " << std::setw(8) << halo_bw_gbps << " GB/s" << std::endl;
        std::cout << std::endl;

        std::cout << "========================================" << std::endl;
        std::cout << "Communication Analysis" << std::endl;
        std::cout << "========================================" << std::endl;

        double fft_comm_pct = 100.0 * fft_comm_time / total_fft_time;
        double halo_comm_pct = 100.0 * global_halo_avg / (global_halo_avg + global_sten_avg);

        std::cout << "FFT comm fraction:   " << std::setw(6) << fft_comm_pct << "%" << std::endl;
        std::cout << "Halo comm fraction:  " << std::setw(6) << halo_comm_pct << "%" << std::endl;
        std::cout << std::endl;

        // RK45 cycle estimate (6 stages, each with demag FFT + exchange + stencil)
        double rk45_fft_time = 6 * total_fft_time;
        double rk45_halo_time = 6 * global_halo_avg;
        double rk45_stencil_time = 6 * global_sten_avg;
        double rk45_total = rk45_fft_time + rk45_halo_time + rk45_stencil_time;

        std::cout << "========================================" << std::endl;
        std::cout << "RK45 Timestep Estimate (6 stages)" << std::endl;
        std::cout << "========================================" << std::endl;
        std::cout << "FFT (demag):     " << std::setw(8) << rk45_fft_time << " ms" << std::endl;
        std::cout << "Halo exchange:   " << std::setw(8) << rk45_halo_time << " ms" << std::endl;
        std::cout << "Stencil ops:     " << std::setw(8) << rk45_stencil_time << " ms" << std::endl;
        std::cout << "Total per step:  " << std::setw(8) << rk45_total << " ms" << std::endl;
        std::cout << "Steps per sec:   " << std::setw(8) << (1000.0 / rk45_total) << std::endl;
        std::cout << std::endl;

        // Memory estimate
        double mem_per_rank_mb = (3 * real_size * sizeof(float) +
                                  complex_size * sizeof(std::complex<float>) +
                                  padded_size * sizeof(float)) / 1e6;
        std::cout << "========================================" << std::endl;
        std::cout << "Memory Usage" << std::endl;
        std::cout << "========================================" << std::endl;
        std::cout << "Per-rank allocation: " << std::setw(8) << mem_per_rank_mb << " MB" << std::endl;
        std::cout << "Total (all ranks):   " << std::setw(8) << mem_per_rank_mb * num_ranks << " MB" << std::endl;
        std::cout << std::endl;

        std::cout << "[BENCHMARK COMPLETE]" << std::endl;
        std::cout << "========================================" << std::endl;
    }

    // Cleanup
    CUDA_CHECK(cudaFree(d_fft_input));
    CUDA_CHECK(cudaFree(d_fft_output));
    CUDA_CHECK(cudaFree(d_freq));
    CUDA_CHECK(cudaFree(d_padded_field));
    CUDA_CHECK(cudaFree(d_stencil_output));

    MPI_Finalize();
    return 0;
}
