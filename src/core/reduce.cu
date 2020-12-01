#include <stdexcept>

#include "cudaerror.hpp"
#include "cudalaunch.hpp"
#include "cudastream.hpp"
#include "datatypes.hpp"
#include "field.hpp"
#include "gpubuffer.hpp"
#include "reduce.hpp"

__global__ void k_maxAbsValue(real* result, CuField f) {
  // Reduce to a block
  __shared__ real sdata[BLOCKDIM];
  int ncells = f.system.grid.ncells();
  int tid = threadIdx.x;
  real threadValue = 0.0;
  for (int i = tid; i < ncells; i += BLOCKDIM) {
    if (!f.cellInGeometry(i))
      continue;
    for (int c = 0; c < f.ncomp; c++) {
      real value = abs(f.valueAt(i, c));
      threadValue = value > threadValue ? value : threadValue;
    }
  }
  sdata[tid] = threadValue;
  __syncthreads();

  // Reduce the block
  for (unsigned int s = BLOCKDIM / 2; s > 0; s >>= 1) {
    if (tid < s)
      if (sdata[tid + s] > sdata[tid])
        sdata[tid] = sdata[tid + s];
    __syncthreads();
  }
  // TODO: check if loop unrolling makes sense here

  // Set the result
  if (tid == 0)
    *result = sdata[0];
}

real maxAbsValue(const Field& f) {
  GpuBuffer<real> d_result(1);
  cudaLaunchReductionKernel(k_maxAbsValue, d_result.get(), f.cu());

  // copy the result to the host and return
  real result;
  checkCudaError(cudaMemcpyAsync(&result, d_result.get(), 1 * sizeof(real),
                                 cudaMemcpyDeviceToHost, getCudaStream()));
  return result;
}

__global__ void k_maxVecNorm(real* result, CuField f) {
  // Reduce to a block
  __shared__ real sdata[BLOCKDIM];
  int ncells = f.system.grid.ncells();
  int tid = threadIdx.x;
  real threadValue = 0.0;
  for (int i = tid; i < ncells; i += BLOCKDIM) {
    if (!f.cellInGeometry(i))
      continue;
    real3 cellVec = f.vectorAt(i);
    real cellNorm = norm(cellVec);
    if (cellNorm > threadValue) {
      threadValue = cellNorm;
    }
  }
  sdata[tid] = threadValue;
  __syncthreads();

  // Reduce the block
  for (unsigned int s = BLOCKDIM / 2; s > 0; s >>= 1) {
    if (tid < s)
      if (sdata[tid + s] > sdata[tid])
        sdata[tid] = sdata[tid + s];
    __syncthreads();
  }
  // TODO: check if loop unrolling makes sense here

  // Set the result
  if (tid == 0)
    *result = sdata[0];
}

real maxVecNorm(const Field& f) {
  if (f.ncomp() != 3) {
    throw std::runtime_error(
        "the input field of maxVecNorm should have 3 components");
  }

  GpuBuffer<real> d_result(1);
  cudaLaunchReductionKernel(k_maxVecNorm, d_result.get(), f.cu());

  // copy the result to the host and return
  real result;
  checkCudaError(cudaMemcpyAsync(&result, d_result.get(), 1 * sizeof(real),
                                 cudaMemcpyDeviceToHost, getCudaStream()));
  return result;
}

__global__ void k_average(real* result, CuField f, int comp) {
  __shared__ real sdata[BLOCKDIM];
  int tid = threadIdx.x;
  int ncells = f.system.grid.ncells();

  // Reduce to a block
  real threadValue = 0.0;
  for (int i = tid; i < ncells; i += BLOCKDIM) {
    if (!f.cellInGeometry(i))
      continue;
    threadValue += f.valueAt(i, comp);
  }
  sdata[tid] = threadValue;
  __syncthreads();

  // Reduce the block
  for (unsigned int s = BLOCKDIM / 2; s > 0; s >>= 1) {
    if (tid < s)
      sdata[tid] += sdata[tid + s];
    __syncthreads();
  }
  // TODO: check if loop unrolling makes sense here

  // Set the result
  if (tid == 0)
    *result = sdata[0] / ncells;
}

real fieldComponentAverage(const Field& f, int comp) {
  if (comp >= f.ncomp()) {
    throw std::runtime_error("Can not take the average of component " +
                             std::to_string(comp) +
                             " of a field which has only " +
                             std::to_string(f.ncomp()) + " components");
  }

  real result;
  GpuBuffer<real> d_result(1);
  cudaLaunchReductionKernel(k_average, d_result.get(), f.cu(), comp);
  checkCudaError(cudaMemcpyAsync(&result, d_result.get(), sizeof(real),
                                 cudaMemcpyDeviceToHost, getCudaStream()));
  return result;
}

std::vector<real> fieldAverage(const Field& f) {
  std::vector<real> result;
  for (int c = 0; c < f.ncomp(); c++)
    result.push_back(fieldComponentAverage(f, c));
  return result;
}

__global__ void k_dotSum(real* result, CuField f, CuField g) {
  __shared__ real sdata[BLOCKDIM];
  int ncells = f.system.grid.ncells();
  int tid = threadIdx.x;

  real threadValue = 0.0;
  for (int i = tid; i < ncells; i += BLOCKDIM) {
    if (!f.cellInGeometry(i))
      continue;

    threadValue += dot(f.vectorAt(i), g.vectorAt(i));
  }
  sdata[tid] = threadValue;
  __syncthreads();

  // Reduce the block
  for (unsigned int s = BLOCKDIM / 2; s > 0; s >>= 1) {
    if (tid < s)
      sdata[tid] += sdata[tid + s];
    __syncthreads();
  }
  // TODO: check if loop unrolling makes sense here

  // Set the result
  if (tid == 0)
    *result = sdata[0];
}

real dotSum(const Field& f, const Field& g) {
  if (f.system() != g.system())
    throw std::invalid_argument(
        "Can not take the dot sum of the two fields because they are not "
        "defined on the same system.");

  GpuBuffer<real> d_result(1);
  cudaLaunchReductionKernel(k_dotSum, d_result.get(), f.cu(), g.cu());

  // copy the result to the host and return
  real result;
  checkCudaError(cudaMemcpyAsync(&result, d_result.get(), sizeof(real),
                                 cudaMemcpyDeviceToHost, getCudaStream()));
  return result;
}
