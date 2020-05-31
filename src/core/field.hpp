#pragma once

#include <iostream>
#include <memory>
#include <vector>

#include "datatypes.hpp"
#include "grid.hpp"

class CuField;
class FieldQuantity;

class Field {
  int ncomp_;
  Grid grid_;
  std::vector<real*> devptrs_;
  real** devptr_devptrs_;

 public:
  Field();
  Field(Grid grid, int nComponents);

  ~Field();

  /// Copy constructor (data on gpu is copied)
  Field(const Field&);

  /// Move constructer
  Field(Field&& other);

  /// Assignment operator (data on gpu is copied)
  Field& operator=(const Field& other);

  /// Evaluates quantity and sets the result in this field
  Field& operator=(const FieldQuantity& other);

  /// Move assignment operator
  Field& operator=(Field&& other);

  Grid grid() const;
  int ncomp() const;
  real* devptr(int comp) const;

  void getData(real* buffer) const;
  void setData(real* buffer);
  void setUniformComponent(real value, int comp);
  void makeZero();

  void copyFrom(const Field*);

  CuField cu() const;

  void operator+=(const Field& x);

 private:
  void allocate();
  void free();
};

struct CuField {
  const Grid grid;
  const int ncomp;
  real** ptrs;

  __device__ bool cellInGrid(int) const;
  __device__ bool cellInGrid(int3) const;

  __device__ real valueAt(int idx, int comp = 0) const;
  __device__ real valueAt(int3 coo, int comp = 0) const;

  __device__ real3 vectorAt(int idx) const;
  __device__ real3 vectorAt(int3 coo) const;

  __device__ void setValueInCell(int idx, int comp, real value);
  __device__ void setVectorInCell(int idx, real3 vec);
};

__device__ inline bool CuField::cellInGrid(int idx) const {
  return grid.cellInGrid(idx);
}

__device__ inline bool CuField::cellInGrid(int3 coo) const {
  return grid.cellInGrid(coo);
}

__device__ inline real CuField::valueAt(int idx, int comp) const {
  return ptrs[comp][idx];
}

__device__ inline real CuField::valueAt(int3 coo, int comp) const {
  return valueAt(grid.coord2index(coo), comp);
}

__device__ inline real3 CuField::vectorAt(int idx) const {
  return real3{ptrs[0][idx], ptrs[1][idx], ptrs[2][idx]};
}

__device__ inline real3 CuField::vectorAt(int3 coo) const {
  return vectorAt(grid.coord2index(coo));
}

__device__ inline void CuField::setValueInCell(int idx, int comp, real value) {
  ptrs[comp][idx] = value;
}

__device__ inline void CuField::setVectorInCell(int idx, real3 vec) {
  ptrs[0][idx] = vec.x;
  ptrs[1][idx] = vec.y;
  ptrs[2][idx] = vec.z;
}