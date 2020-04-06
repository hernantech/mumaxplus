#include <vector>

#include "cudalaunch.hpp"
#include "field.hpp"
#include "fieldops.hpp"

__global__ void k_addFields(CuField y,
                            real a1,
                            CuField x1,
                            real a2,
                            CuField x2) {
  if (!y.cellInGrid())
    return;
  int nComp = y.ncomp;
  for (int c = -0; c < nComp; c++) {
    real term1 = a1 * x1.cellValue(c);
    real term2 = a2 * x2.cellValue(c);
    y.setCellValue(c, term1 + term2);
  }
}

// TODO: throw error if grids or number of components do not match
void add(Field* y, real a1, const Field* x1, real a2, const Field* x2) {
  int ncells = y->grid().ncells();
  cudaLaunch(ncells, k_addFields, y->cu(), a1, x1->cu(), a2, x2->cu());
}

void add(Field* y, const Field* x1, const Field* x2) {
  add(y, 1, x1, 1, x2);
}

// TODO: this can be done much more efficient
void add(Field* y, std::vector<const Field*> x, std::vector<real> weights) {
  // TODO:: throw error if inputs are not compatible
  if (x.size() == 1) {
    add(y, 0, x.at(0), weights.at(0), x.at(0));
  }

  add(y, weights.at(0), x.at(0), weights.at(1), x.at(1));
  for (int n = 2; n < x.size(); n++) {
    if (weights.at(n) != 0.0) {
      add(y, 1, y, weights.at(n), x.at(n));
    }
  }
}

__global__ void k_normalize(CuField dst, CuField src) {
  if (!dst.cellInGrid())
    return;
  int nComp = src.ncomp;
  real* values = new real[nComp];
  real norm2 = 0.0;
  for (int c = 0; c < nComp; c++) {
    values[c] = src.cellValue(c);
    norm2 += values[c] * values[c];
  }
  real invnorm = rsqrt(norm2);
  for (int c = 0; c < nComp; c++) {
    dst.setCellValue(c, values[c] * invnorm);
  }
  delete values;
}

void normalized(Field* dst, const Field* src) {
  // TODO: check field dimensions
  cudaLaunch(dst->grid().ncells(), k_normalize, dst->cu(), src->cu());
}

std::unique_ptr<Field> normalized(const Field* src) {
  std::unique_ptr<Field> dst(new Field(src->grid(), src->ncomp()));
  normalized(dst.get(), src);
  return dst;
}

void normalize(Field* f) {
  normalized(f, f);
}