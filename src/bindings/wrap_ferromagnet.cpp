#include <memory>
#include <stdexcept>

#include "afmexchange.hpp"
#include "anisotropy.hpp"
#include "conductivitytensor.hpp"
#include "demag.hpp"
#include "dmi.hpp"
#include "effectivefield.hpp"
#include "electricalpotential.hpp"
#include "energy.hpp"
#include "exchange.hpp"
#include "ferromagnet.hpp"
#include "fieldquantity.hpp"
#include "fullmag.hpp"
#include "magnet.hpp"
#include "mumaxworld.hpp"
#include "parameter.hpp"
#include "stt.hpp"
#include "thermalnoise.hpp"
#include "torque.hpp"
#include "world.hpp"
#include "wrappers.hpp"
#include "zeeman.hpp"

void wrap_ferromagnet(py::module& m) {
  py::class_<Ferromagnet, Magnet>(m, "Ferromagnet")
      .def_property_readonly("magnetization", &Ferromagnet::magnetization)

      .def_readwrite("enable_demag", &Ferromagnet::enableDemag)
      .def_readwrite("enable_openbc", &Ferromagnet::enableOpenBC)
      .def_readwrite("enable_zhang_li_torque", &Ferromagnet::enableZhangLiTorque)
      .def_readwrite("enable_slonczewski_torque", &Ferromagnet::enableSlonczewskiTorque)
      .def_readwrite("bias_magnetic_field", &Ferromagnet::biasMagneticField,
                     "uniform external magnetic field")

      .def_readonly("msat", &Ferromagnet::msat)
      .def_readonly("alpha", &Ferromagnet::alpha)
      .def_readonly("aex", &Ferromagnet::aex)
      .def_readonly("ku1", &Ferromagnet::ku1)
      .def_readonly("ku2", &Ferromagnet::ku2)
      .def_readonly("kc1", &Ferromagnet::kc1)
      .def_readonly("kc2", &Ferromagnet::kc2)
      .def_readonly("kc3", &Ferromagnet::kc3)
      .def_readonly("anisU", &Ferromagnet::anisU)
      .def_readonly("anisC1", &Ferromagnet::anisC1)
      .def_readonly("anisC2", &Ferromagnet::anisC2)
      .def_readonly("Lambda", &Ferromagnet::Lambda)
      .def_readonly("FreeLayerThickness", &Ferromagnet::FreeLayerThickness)
      .def_readonly("eps_prime", &Ferromagnet::eps_prime)
      .def_readonly("FixedLayer", &Ferromagnet::FixedLayer)
      .def_readonly("xi", &Ferromagnet::xi)
      .def_readonly("pol", &Ferromagnet::pol)
      .def_readonly("jcur", &Ferromagnet::jcur)
      .def_readonly("temperature", &Ferromagnet::temperature)
      .def_readonly("dmi_tensor", &Ferromagnet::dmiTensor)
      .def_readonly("applied_potential", &Ferromagnet::appliedPotential)
      .def_readonly("conductivity", &Ferromagnet::conductivity)
      .def_readonly("amr_ratio", &Ferromagnet::amrRatio)
      .def_readwrite("RelaxTorqueThreshold", &Ferromagnet::RelaxTorqueThreshold)
      .def_readonly("poisson_system", &Ferromagnet::poissonSystem)
      
      .def("minimize", &Ferromagnet::minimize, py::arg("tol"), py::arg("nsamples"))
      .def("relax", &Ferromagnet::relax, py::arg("tol"));

  m.def("torque", &torqueQuantity);
  m.def("llg_torque", &llgTorqueQuantity);
  m.def("damping_torque", &relaxTorqueQuantity);
  m.def("spin_transfer_torque", &spinTransferTorqueQuantity);
  m.def("max_torque", &maxTorqueQuantity);

  m.def("demag_field", &demagFieldQuantity);
  m.def("demag_energy_density", &demagEnergyDensityQuantity);
  m.def("demag_energy", &demagEnergyQuantity);

  m.def("anisotropy_field", &anisotropyFieldQuantity);
  m.def("anisotropy_energy_density", &anisotropyEnergyDensityQuantity);
  m.def("anisotropy_energy", &anisotropyEnergyQuantity);

  // normal Ferromagnet
  m.def("exchange_field", &exchangeFieldQuantity);
  m.def("exchange_energy_density", &exchangeEnergyDensityQuantity);
  m.def("exchange_energy", &exchangeEnergyQuantity);  
  m.def("max_angle", py::overload_cast<const Ferromagnet*>(&maxAngle));
  // ferromagnetic sublattice
  m.def("inhomogeneous_exchange_field", &inHomoAfmExchangeFieldQuantity);
  m.def("homogeneous_exchange_field", &homoAfmExchangeFieldQuantity);
  m.def("inhomogeneous_exchange_energy_density", &inHomoAfmExchangeEnergyDensityQuantity);
  m.def("homogeneous_exchange_energy_density", &homoAfmExchangeEnergyDensityQuantity);
  m.def("inhomogeneous_exchange_energy", &inHomoAfmExchangeEnergyQuantity);
  m.def("homogeneous_exchange_energy", &homoAfmExchangeEnergyQuantity);
  //
  m.def("dmi_field", &dmiFieldQuantity);
  m.def("dmi_energy_density", &dmiEnergyDensityQuantity);
  m.def("dmi_energy", &dmiEnergyQuantity);

  m.def("external_field", &externalFieldQuantity);
  m.def("zeeman_energy_density", &zeemanEnergyDensityQuantity);
  m.def("zeeman_energy", &zeemanEnergyQuantity);

  m.def("effective_field", &effectiveFieldQuantity);
  m.def("total_energy_density", &totalEnergyDensityQuantity);
  m.def("total_energy", &totalEnergyQuantity);

  m.def("conductivity_tensor", &conductivityTensorQuantity);
  m.def("electrical_potential", &electricalPotentialQuantity);

  m.def("thermal_noise", &thermalNoiseQuantity);

  m.def("full_magnetization",
        py::overload_cast<const Ferromagnet*>(&fullMagnetizationQuantity));
}