#include "cudalaunch.hpp"
#include "inter_parameter.hpp"
#include "reduce.hpp"

#include <algorithm>

InterParameter::InterParameter(std::shared_ptr<const System> system, real value)
    : system_(system),
      uniqueRegions_(GpuBuffer<uint>(system->uniqueRegions)) {
        int N = *std::max_element(system->uniqueRegions.begin(),
                                  system->uniqueRegions.end()
                                  ) + 1;
        valuesbuffer_ = GpuBuffer<real>(std::vector<real>(N * (N - 1) / 2, value));
        numRegions_ = system->uniqueRegions.size();
      }

GpuBuffer<uint> InterParameter::uniqueRegions() const {
    return uniqueRegions_;
}

const GpuBuffer<real>& InterParameter::values() const {
    return valuesbuffer_;
}

size_t InterParameter::numberOfRegions() const {
    return numRegions_;
}

__global__ void k_set(real* values, real value) {
    // TODO: more efficient, memory-wise, to only set relevant elements?
    int idx = blockIdx.x * blockDim.x + threadIdx.x;
    values[idx] = value;
}

void InterParameter::set(real value) {
    uint N = numRegions_ * (numRegions_ - 1) / 2;
    cudaLaunch(N, k_set, valuesbuffer_.get(), value);
}

__global__ void k_setBetween(real* values, int index, real value) {
    values[index] = value;
}

void InterParameter::setBetween(uint i, uint j, real value) {
    system_->checkIdxInRegions(i);
    system_->checkIdxInRegions(j);

    // Update GPU memory directly
    cudaLaunchReductionKernel(k_setBetween, valuesbuffer_.get(), getLutIndex(i, j), value);
    return;
}

CuInterParameter InterParameter::cu() const {
    return CuInterParameter(this);
}