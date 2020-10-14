#include <memory>
#include <vector>

#include "dynamicequation.hpp"
#include "ferromagnetquantity.hpp"
#include "field.hpp"
#include "fieldquantity.hpp"
#include "timesolver.hpp"
#include "variable.hpp"
#include "wrappers.hpp"

void wrap_timesolver(py::module& m) {
  py::class_<TimeSolver>(m, "TimeSolver")
      .def(py::init([](Variable* x, FM_FieldQuantity rhs) {
             return std::unique_ptr<TimeSolver>(new TimeSolver(DynamicEquation(
                 x, std::unique_ptr<FieldQuantity>(rhs.clone()))));
           }),
           py::arg("variable"), py::arg("rhs"))
      //.def(py::init([](Variable* x, FieldQuantity* rhs, FieldQuantity * noise)
      //{
      //       return std::unique_ptr<TimeSolver>(
      //           new TimeSolver(DynamicEquation(x, rhs, noise)));
      //     }),
      //     py::arg("variable"), py::arg("rhs"), py::arg("noiseterm"))
      //.def(py::init(
      //         [](std::vector<std::pair<Variable*, FieldQuantity*>> eqPairs) {
      //           std::vector<DynamicEquation> eqs;
      //           for (const auto& eqPair : eqPairs)
      //             eqs.emplace_back(eqPair.first, eqPair.second);
      //           return std::unique_ptr<TimeSolver>(new TimeSolver(eqs));
      //         }))
      .def_property_readonly("time", &TimeSolver::time)
      .def("step", &TimeSolver::step)
      .def("steps", &TimeSolver::steps)
      .def_property("timestep", &TimeSolver::timestep, &TimeSolver::setTimeStep)
      .def_property("adaptive_timestep", &TimeSolver::adaptiveTimeStep,
                    [](TimeSolver& solver, bool adaptive) {
                      if (adaptive) {
                        solver.enableAdaptiveTimeStep();
                      } else {
                        solver.disableAdaptiveTimeStep();
                      }
                    })
      .def("run", &TimeSolver::run);
}