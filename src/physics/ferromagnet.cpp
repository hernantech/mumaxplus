#include "ferromagnet.hpp"

#include <random>

#include "minimizer.hpp"

Ferromagnet::Ferromagnet(World* world, std::string name, Grid grid)
    : System(world, name, grid),
      demagField_(this),
      anisotropyField_(this),
      exchangeField_(this),
      effectiveField_(this),
      torque_(this),
      magnetization_(name + ":magnetization", "", 3, grid),
      ku1(grid, 0.0),
      alpha(grid, 0.0),
      anisU(grid, {0,0,0}) {
  aex = 0;
  msat = 1.0;
  enableDemag = true;
  {
    // TODO: this can be done much more efficient somewhere else
    int ncomp = 3;
    int nvalues = ncomp * grid_.ncells();
    std::vector<real> randomValues(nvalues);
    std::uniform_real_distribution<real> unif(-1, 1);
    std::default_random_engine randomEngine;
    for (auto& v : randomValues) {
      v = unif(randomEngine);
    }
    Field randomField(grid_, ncomp);
    randomField.setData(&randomValues[0]);
    magnetization_.set(&randomField);
  }
}

Ferromagnet::~Ferromagnet() {}

const Variable* Ferromagnet::magnetization() const {
  return &magnetization_;
}

const Quantity* Ferromagnet::demagField() const {
  return &demagField_;
}

const Quantity* Ferromagnet::anisotropyField() const {
  return &anisotropyField_;
}

const Quantity* Ferromagnet::exchangeField() const {
  return &exchangeField_;
}

const Quantity* Ferromagnet::effectiveField() const {
  return &effectiveField_;
}

const Quantity* Ferromagnet::torque() const {
  return &torque_;
}

void Ferromagnet::minimize(real tol, int nSamples) {
  Minimizer minimizer(this, tol, nSamples);
  minimizer.exec();
}