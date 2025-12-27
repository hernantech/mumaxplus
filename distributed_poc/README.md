# mumax+ Distributed Multi-GPU Proof-of-Concept

This directory contains proof-of-concept implementations for porting mumax+ to multi-GPU using MPI + HeFFTe.

## Overview

The multi-GPU architecture uses:
- **Slab decomposition** along the Z-axis (slowest dimension)
- **HeFFTe** for distributed 3D FFT (demagnetization field)
- **CUDA-aware MPI** for halo exchange (stencil operations)
- **MPI_Allreduce** for global reductions (adaptive timestepping)

## Prerequisites

### Required Software
- CUDA Toolkit 11.0+
- MPI implementation (OpenMPI or MPICH)
- CMake 3.18+
- HeFFTe library with CUDA support

### Installing HeFFTe

```bash
# Clone HeFFTe
git clone https://github.com/icl-utk-edu/heffte.git
cd heffte
mkdir build && cd build

# Configure with CUDA and MPI support
cmake -DCMAKE_INSTALL_PREFIX=/opt/heffte \
      -DHeffte_ENABLE_CUDA=ON \
      -DHeffte_ENABLE_MPI=ON \
      -DHeffte_ENABLE_FFTW=OFF \
      -DCMAKE_CUDA_ARCHITECTURES="70;80;86" \
      ..

# Build and install
make -j$(nproc)
sudo make install
```

### CUDA Architectures
Adjust `CMAKE_CUDA_ARCHITECTURES` for your GPU:
- `70` = V100
- `80` = A100
- `86` = RTX 3090
- `89` = RTX 4090

## Building the PoC

```bash
cd distributed_poc
mkdir build && cd build

# Configure (specify HeFFTe location if not in standard path)
cmake -DHEFFTE_DIR=/opt/heffte ..

# Build
make -j$(nproc)
```

## Running Tests

### HeFFTe Distributed FFT Test
```bash
# 2 GPUs
mpirun -np 2 ./heffte_poc

# 4 GPUs
mpirun -np 4 ./heffte_poc
```

Expected output:
```
========================================
HeFFTe Distributed FFT Proof-of-Concept
========================================
MPI Ranks: 2
CUDA Devices: 2
Grid Size: 128 x 128 x 128
...
Max Error: 1.23456e-07
[SUCCESS] Distributed FFT convolution validated!
========================================
```

### Halo Exchange Test
```bash
mpirun -np 2 ./halo_exchange_test
mpirun -np 4 ./halo_exchange_test
```

Expected output:
```
========================================
Halo Exchange Test for mumax+ Port
========================================
...
Global Max Halo Error: 0
[SUCCESS] Halo exchange validated!
========================================
```

## Architecture Details

### File Descriptions

| File | Purpose |
|------|---------|
| `heffte_poc.cu` | Validates distributed FFT convolution (demagnetization) |
| `halo_exchange_test.cu` | Validates MPI halo exchange for stencil operations |
| `validation_test.cu` | Compares multi-GPU HeFFTe vs single-GPU cuFFT results |
| `benchmark_test.cu` | Measures communication overhead and scaling |
| `CMakeLists.txt` | Build configuration |

### Data Layout

Fields are stored in padded buffers to accommodate halos:

```
Buffer layout for one Z-slab:
┌─────────────────────────────────────────────────────────────┐
│ Lower Halo │        Real Data        │ Upper Halo │
│ (1 plane)  │     (local_Nz planes)   │ (1 plane)  │
└─────────────────────────────────────────────────────────────┘
     ↑                   ↑                    ↑
  Received           Owned by            Received
  from rank-1        this rank           from rank+1
```

### Communication Patterns

1. **Halo Exchange** (before stencil operations):
   - Point-to-point MPI between neighboring ranks
   - Only boundary planes are exchanged

2. **Distributed FFT** (for demagnetization):
   - HeFFTe handles internal AllToAll transposes
   - Each rank processes its local frequency chunk

3. **Global Reductions** (for adaptive timestepping):
   - MPI_Allreduce on scalars (max error)
   - Ensures all ranks use same timestep

## Benchmark Results

Tested on 2x NVIDIA RTX A5000 GPUs with CUDA-aware OpenMPI 5.0.6.

### 128³ Grid (2 GPUs)

| Operation | Time (ms) | Notes |
|-----------|-----------|-------|
| Forward FFT | 8.5 | Includes AllToAll transpose |
| Backward FFT | 8.7 | Includes AllToAll transpose |
| Kernel Apply | 0.02 | Negligible |
| **FFT Cycle Total** | **17.3** | |
| Halo Exchange | 0.37 | 0.26 MB per rank |
| Stencil Compute | 0.03 | |

**Communication overhead**: FFT ~99.9%, Halo ~92%

**RK45 timestep estimate**: 106 ms (9.4 steps/sec)

### 256³ Grid (2 GPUs)

| Operation | Time (ms) | Notes |
|-----------|-----------|-------|
| Forward FFT | 60.6 | Includes AllToAll transpose |
| Backward FFT | 61.3 | Includes AllToAll transpose |
| Kernel Apply | 0.12 | Negligible |
| **FFT Cycle Total** | **121.9** | |
| Halo Exchange | 2.15 | 1.05 MB per rank |
| Stencil Compute | 0.16 | |

**Communication overhead**: FFT ~99.9%, Halo ~93%

**RK45 timestep estimate**: 745 ms (1.3 steps/sec)

### Key Findings

1. **FFT dominates**: Communication (AllToAll transpose) is ~99.9% of FFT time
2. **Halo exchange is fast**: <3ms even for 256³ grids
3. **Memory per rank**: ~21 MB (128³) to ~169 MB (256³)
4. **Validation**: Multi-GPU produces identical results to single-GPU (zero error)

## Integration Roadmap

### Phase 1: Validate PoC (This directory) ✅ Complete
- [x] HeFFTe distributed FFT
- [x] Halo exchange mechanism
- [x] Multi-GPU vs single-GPU validation (zero error)
- [x] Benchmark communication overhead

### Phase 2: Infrastructure Classes ✅ Complete
Located in `src/distributed/`:

| File | Description |
|------|-------------|
| `mpicontext.hpp/cu` | MPI singleton - initialization, GPU binding, Z-slab decomposition |
| `distributedgrid.hpp/cu` | Grid management - local/global coordinate mapping, halo padding |
| `haloexchanger.hpp/cu` | MPI halo exchange - sync/async, CUDA-aware MPI detection |
| `test_infrastructure.cu` | Validation test for all infrastructure classes |
| `CMakeLists.txt` | Build configuration with MPI/HeFFTe integration |

**Build & Test:**
```bash
cd mumaxplus
mkdir build-distributed && cd build-distributed
cmake -DENABLE_DISTRIBUTED=ON -DHEFFTE_DIR=$HOME/heffte -DCMAKE_CUDA_ARCHITECTURES=86 ..
make test_infrastructure
mpirun -np 2 ./src/distributed/test_infrastructure
```

**Test Results (2x RTX A5000):**
```
===================================
Distributed Infrastructure Tests
Grid: 64x64x32, Ranks: 2
===================================
Test 1: MPIContext      - PASS
Test 2: DistributedGrid - PASS
Test 3: HaloExchanger   - PASS
===================================
All tests PASSED
```

### Phase 3: Integrate with mumax+
- [ ] Replace `StrayFieldFFTExecutor` with HeFFTe
- [ ] Update stencil kernels for padded buffers
- [ ] Add MPI_Allreduce to reduction functions

### Phase 4: Validate Physics
- [ ] Compare against single-GPU results
- [ ] Run Standard Problem #4
- [ ] Benchmark scaling efficiency

## Troubleshooting

### HeFFTe not found
```
cmake -DHEFFTE_DIR=/path/to/heffte/install ..
```

### CUDA-aware MPI issues
Ensure MPI is built with CUDA support:
```bash
ompi_info --parsable --all | grep cuda
```

### Out of GPU memory
Reduce grid size or use more GPUs:
```bash
# In heffte_poc.cu, modify:
const int Nx = 64;  // Reduce from 128
```

### Rank assignment issues
Use explicit device binding:
```bash
mpirun -np 4 --map-by ppr:1:gpu ./heffte_poc
```

### CUDA 11.x + GCC 11 compilation errors
If you see `std::function` parameter pack errors, install g++-10:
```bash
sudo apt install g++-10
```
CMake will automatically use it as the CUDA host compiler.

## Performance Tips

1. **Use GPUDirect RDMA** if available (InfiniBand + NVIDIA GPUs)
2. **Pin memory** for MPI staging buffers (already done in halo_exchange_test)
3. **Overlap communication** with interior computation
4. **Use power-of-2 grid sizes** for optimal FFT performance

## References

- [HeFFTe Documentation](https://icl-utk-edu.github.io/heffte/)
- [mumax+ Repository](https://github.com/mumax/plus)
- [NVIDIA Multi-GPU Programming](https://developer.nvidia.com/blog/introduction-cuda-aware-mpi/)
