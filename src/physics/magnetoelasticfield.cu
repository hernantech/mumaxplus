// TODO: check if these includes are really all necessary
#include "cudalaunch.hpp"
#include "elasticforce.hpp"
#include "energy.hpp"
#include "ferromagnet.hpp"
#include "field.hpp"
#include "parameter.hpp"
#include "reduce.hpp"
#include "straintensor.hpp"
#include "world.hpp"


bool magnetoelasticAssuredZero(const Ferromagnet* magnet) {
  return ((!magnet->getEnableElastodynamics()) ||
          (magnet->msat.assuredZero()) ||
          (magnet->B1.assuredZero() && magnet->B2.assuredZero()));
}

__global__ void k_magnetoelasticField(CuField hField,
                                      const CuField mField,
                                      const CuField strain,
                                      const CuParameter B1,
                                      const CuParameter B2,
                                      const CuParameter msat) {
  const int idx = blockIdx.x * blockDim.x + threadIdx.x;
  const CuSystem system = hField.system;
  const Grid grid = system.grid;
  // When outside the geometry, set to zero and return early
  if (!system.inGeometry(idx)) {
    if (grid.cellInGrid(idx)) {
      hField.setVectorInCell(idx, real3{0, 0, 0});
    }
    return;
  }

  int ip1, ip2;
  for (int i=0; i<3; i++) {
    ip1 = i+1; ip2 = i+2;
    // If they exceed 3, loop around
    if (ip1 >= 3) ip1 -= 3;
    if (ip2 >= 3) ip2 -= 3;

    hField.setValueInCell(idx, i,
          - 2 / msat.valueAt(idx) *
          (B1.valueAt(idx) * strain.valueAt(idx, i) * mField.valueAt(idx, i) + 
           B2.valueAt(idx) * (strain.valueAt(idx, i+ip1+2) * mField.valueAt(idx, ip1) + 
                              strain.valueAt(idx, i+ip2+2) * mField.valueAt(idx, ip2))));
  }
}

Field evalMagnetoelasticField(const Ferromagnet* magnet) {
  Field hField(magnet->system(), 3);
  if (magnetoelasticAssuredZero(magnet)) {
    hField.makeZero();
    return hField;
  }

  int ncells = hField.grid().ncells();
  CuField mField = magnet->magnetization()->field().cu();
  CuField strain = evalStrainTensor(magnet).cu();
  CuParameter B1 = magnet->B1.cu();
  CuParameter B2 = magnet->B2.cu();
  CuParameter msat = magnet->msat.cu();

  cudaLaunch(ncells, k_magnetoelasticField, hField.cu(), mField, strain, B1, B2, msat);
  return hField;
}


FM_FieldQuantity magnetoelasticFieldQuantity(const Ferromagnet* magnet) {
  return FM_FieldQuantity(magnet, evalMagnetoelasticField, 3,
                          "magnetoelastic_field", "T");
}
