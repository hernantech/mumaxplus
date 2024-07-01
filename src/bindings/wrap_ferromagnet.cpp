#include <memory>
#include <stdexcept>

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
#include "mumaxworld.hpp"
#include "parameter.hpp"
#include "strayfieldkernel.hpp"
#include "stt.hpp"
#include "thermalnoise.hpp"
#include "torque.hpp"
#include "world.hpp"
#include "wrappers.hpp"
#include "zeeman.hpp"

void wrap_ferromagnet(py::module& m) {
  py::class_<Ferromagnet>(m, "Ferromagnet")
      .def_property_readonly("name", &Ferromagnet::name)
      .def_property_readonly("system", &Ferromagnet::system)
      .def_property_readonly("world", &Ferromagnet::mumaxWorld)
      .def_property_readonly("magnetization", &Ferromagnet::magnetization)

      .def_readwrite("enable_demag", &Ferromagnet::enableDemag)
      .def_readwrite("enable_openbc", &Ferromagnet::enableOpenBC)
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
      .def_readonly("poisson_system", &Ferromagnet::poissonSystem)
      
      .def(
          "magnetic_field_from_magnet",
          [](const Ferromagnet* fm, Ferromagnet* magnet) {
            const StrayField* strayField = fm->getStrayField(magnet);
            if (!strayField)
              throw std::runtime_error(
                  "Can not compute the magnetic field of the magnet");
            return strayField;
          },
          py::return_value_policy::reference)

      .def("minimize", &Ferromagnet::minimize, py::arg("tol") = 1e-6,
           py::arg("nsamples") = 10);

  m.def("torque", &torqueQuantity);
  m.def("llg_torque", &llgTorqueQuantity);
  m.def("spin_transfer_torque", &spinTransferTorqueQuantity);

  m.def("demag_field", &demagFieldQuantity);
  m.def("demag_energy_density", &demagEnergyDensityQuantity);
  m.def("demag_energy", &demagEnergyQuantity);

  m.def("anisotropy_field", &anisotropyFieldQuantity);
  m.def("anisotropy_energy_density", &anisotropyEnergyDensityQuantity);
  m.def("anisotropy_energy", &anisotropyEnergyQuantity);

  m.def("exchange_field", &exchangeFieldQuantity);
  m.def("exchange_energy_density", &exchangeEnergyDensityQuantity);
  m.def("exchange_energy", &exchangeEnergyQuantity);
  m.def("max_angle", &maxAngle);

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

  m.def("_demag_kernel", [](const Ferromagnet* fm) {
    Grid grid = fm->grid();
    real3 cellsize = fm->world()->cellsize();
    StrayFieldKernel demagKernel(grid, grid, fm->world());
    return fieldToArray(demagKernel.field());
  });
}