#pragma once

#include <vector>

#include "datatypes.hpp"
#include "ferromagnet.hpp"
#include "grid.hpp"

class World {
 public:
  World(real3 cellsize);
  ~World();
  real3 cellsize() const;

  Ferromagnet* addFerromagnet(std::string name, Grid grid);

 private:
  std::vector<Ferromagnet> Ferromagnets;
  real3 cellsize_;
};