#include "cudalaunch.hpp"
#include "elastodynamics.hpp"
#include "magnet.hpp"
#include "field.hpp"
#include "straintensor.hpp"
#include "stresstensor.hpp"


__global__ void k_stressTensor(CuField stressTensor,
                         const CuField strain,
                         const CuParameter c11,
                         const CuParameter c12,
                         const CuParameter c44) {
  const int idx = blockIdx.x * blockDim.x + threadIdx.x;
  const CuSystem system = stressTensor.system;

  // When outside the geometry, set to zero and return early
  if (!system.inGeometry(idx)) {
    if (system.grid.cellInGrid(idx)) {
      for (int i = 0; i < strain.ncomp; i++)
        stressTensor.setValueInCell(idx, i, 0);
    }
    return;
  }

  int ip1, ip2;
  for (int i=0; i<3; i++) {
    ip1 = i+1; ip2 = i+2;
    if (ip1 >= 3) ip1 -= 3;
    if (ip2 >= 3) ip2 -= 3;

    stressTensor.setValueInCell(idx, i,
                               c11.valueAt(idx) * strain.valueAt(idx, i) +
                               c12.valueAt(idx) * strain.valueAt(idx, ip1) +
                               c12.valueAt(idx) * strain.valueAt(idx, ip2));
    
    stressTensor.setValueInCell(idx, i+3, c44.valueAt(idx) * strain.valueAt(idx, i+3));
  }
}

Field evalStressTensor(const Magnet* magnet) {
  Field stressTensor(magnet->system(), 6);
  if (elasticityAssuredZero(magnet)) {
    stressTensor.makeZero();
    return stressTensor;
  }

  int ncells = stressTensor.grid().ncells();
  Field strain = evalStrainTensor(magnet);
  CuParameter c11 = magnet->c11.cu();
  CuParameter c12 = magnet->c12.cu();
  CuParameter c44 = magnet->c44.cu();

  cudaLaunch(ncells, k_stressTensor, stressTensor.cu(), strain.cu(), c11, c12, c44);
  return stressTensor;
}

M_FieldQuantity stressTensorQuantity(const Magnet* magnet) {
  return M_FieldQuantity(magnet, evalStressTensor, 6, "stress_tensor", "N/m2");
}
