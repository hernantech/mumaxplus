// TODO: check if these includes are really all necessary
#include "cudalaunch.hpp"
#include "energy.hpp"
#include "ferromagnet.hpp"
#include "field.hpp"
#include "magnetoelasticfield.hpp"
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

  for (int i=0; i<3; i++) {
    int ip1 = i+1;
    int ip2 = i+2;

    // If they exceed 3, loop around
    if (ip1 >= 3){
      ip1 -= 3;
    } 
    if (ip2 >= 3){
      ip2 -= 3;
    }

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


Field evalMagnetoelasticEnergyDensity(const Ferromagnet* magnet) {
  if (magnetoelasticAssuredZero(magnet))
    return Field(magnet->system(), 1, 0.0);
  return evalEnergyDensity(magnet, evalMagnetoelasticField(magnet), 0.5);
}

real magnetoelasticEnergy(const Ferromagnet* magnet) {
  if (magnetoelasticAssuredZero(magnet))
    return 0.0;

  real edens = magnetoelasticEnergyDensityQuantity(magnet).average()[0];
  int ncells = magnet->grid().ncells();
  real cellVolume = magnet->world()->cellVolume();
  return ncells * edens * cellVolume;
}

FM_FieldQuantity magnetoelasticFieldQuantity(const Ferromagnet* magnet) {
  return FM_FieldQuantity(magnet, evalMagnetoelasticField, 3,
                          "magnetoelastic_field", "T");
}

FM_FieldQuantity magnetoelasticEnergyDensityQuantity(const Ferromagnet* magnet) {
  return FM_FieldQuantity(magnet, evalMagnetoelasticEnergyDensity, 1, "magnetoelastic_energy_density", "J/m3");
}

FM_ScalarQuantity magnetoelasticEnergyQuantity(const Ferromagnet* magnet) {
  return FM_ScalarQuantity(magnet, magnetoelasticEnergy, "magnetoelastic_energy", "J");
}