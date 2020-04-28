#pragma once

#include "ferromagnetquantity.hpp"

class Ferromagnet;
class Field;

class ExchangeField : public FerromagnetFieldQuantity {
 public:
  ExchangeField(Ferromagnet*);
  void evalIn(Field*) const;
};

class ExchangeEnergyDensity : public FerromagnetFieldQuantity {
 public:
  ExchangeEnergyDensity(Ferromagnet*);
  void evalIn(Field*) const;
};

class ExchangeEnergy : public FerromagnetScalarQuantity {
 public:
  ExchangeEnergy(Ferromagnet*);
  real eval() const;
};