#pragma once

#include <cuda_runtime_api.h>
#include <math.h>

#include <iostream>

typedef double real;
typedef double3 real3;

#define __CUDAOP__ inline __device__ __host__

__CUDAOP__ void operator+=(int3& a, const int3& b) {
  a.x += b.x;
  a.y += b.y;
  a.z += b.z;
}

__CUDAOP__ void operator-=(int3& a, const int3& b) {
  a.x -= b.x;
  a.y -= b.y;
  a.z -= b.z;
}

__CUDAOP__ int3 operator+(const int3& a, const int3& b) {
  return int3{a.x + b.x, a.y + b.y, a.z + b.z};
}

__CUDAOP__ int3 operator-(const int3& a) {
  return int3{-a.x, -a.y, -a.z};
}

__CUDAOP__ int3 operator-(const int3& a, const int3& b) {
  return int3{a.x - b.x, a.y - b.y, a.z - b.z};
}

__CUDAOP__ bool operator==(const int3& a, const int3& b) {
  return (a.x == b.x) && (a.y == b.y) && (a.z == b.z);
}

__CUDAOP__ bool operator!=(const int3& a, const int3& b) {
  return (a.x != b.x) || (a.y != b.y) || (a.z != b.z);
}

inline __host__ std::ostream& operator<<(std::ostream& os, const int3 a) {
  os << "(" << a.x << "," << a.y << "," << a.z << ")";
  return os;
}

__CUDAOP__ void operator+=(real3& a, const real3& b) {
  a.x += b.x;
  a.y += b.y;
  a.z += b.z;
}

__CUDAOP__ void operator+=(real3& a, const real& b) {
  a.x += b;
  a.y += b;
  a.z += b;
}

__CUDAOP__ void operator-=(real3& a, const real3& b) {
  a.x -= b.x;
  a.y -= b.y;
  a.z -= b.z;
}

__CUDAOP__ void operator-=(real3& a, const real& b) {
  a.x -= b;
  a.y -= b;
  a.z -= b;
}

__CUDAOP__ void operator*=(real3& a, const real& b) {
  a.x *= b;
  a.y *= b;
  a.z *= b;
}

__CUDAOP__ void operator/=(real3& a, const real& b) {
  a.x /= b;
  a.y /= b;
  a.z /= b;
}

__CUDAOP__ real3 operator+(const real3& a, const real3& b) {
  return real3{a.x + b.x, a.y + b.y, a.z + b.z};
}

__CUDAOP__ real3 operator+(const real& a, const real3& b) {
  return real3{a + b.x, a + b.y, a + b.z};
}

__CUDAOP__ real3 operator+(const real3& a, const real& b) {
  return real3{a.x + b, a.y + b, a.z + b};
}

__CUDAOP__ real3 operator-(const real3& a) {
  return real3{-a.x, -a.y, -a.z};
}

__CUDAOP__ real3 operator-(const real3& a, const real3& b) {
  return real3{a.x - b.x, a.y - b.y, a.z - b.z};
}

__CUDAOP__ real3 operator-(const real3& a, const real& b) {
  return real3{a.x - b, a.y - b, a.z - b};
}

__CUDAOP__ real3 operator-(const real& a, const real3& b) {
  return real3{a - b.x, a - b.y, a - b.z};
}

__CUDAOP__ real3 operator*(const real& a, const real3& b) {
  return real3{a * b.x, a * b.y, a * b.z};
}

__CUDAOP__ real3 operator*(const real3& a, const real& b) {
  return real3{a.x * b, a.y * b, a.z * b};
}

__CUDAOP__ real3 operator/(const real3& a, const real& b) {
  return real3{a.x / b, a.y / b, a.z / b};
}

__CUDAOP__ real dot(const real3& a, const real3& b) {
  return a.x * b.x + a.y * b.y + a.z * b.z;
}

__CUDAOP__ real3 cross(const real3& a, const real3& b) {
  return real3{a.y * b.z - a.z * b.y, a.z * b.x - a.x * b.z,
               a.x * b.y - a.y * b.x};
}

__CUDAOP__ real norm(const real3& a) {
  return sqrt(dot(a, a));
}

inline __host__ std::ostream& operator<<(std::ostream& os, const real3 a) {
  os << "(" << a.x << "," << a.y << "," << a.z << ")";
  return os;
}