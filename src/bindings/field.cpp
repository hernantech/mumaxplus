#include<pybind11/numpy.h>
#include<stdexcept>

#include"wrappers.hpp"
#include"field.hpp"

void wrap_field(py::module& m) {

    py::class_<Field>(m, "Field")
        .def(py::init<Grid, int>())
        .def("grid", &Field::grid)
        .def("nComponents", &Field::ncomp)
        .def("get", [](const Field* f) {return fieldToArray(f);})
        .def("set", [](Field* f, py::array_t<real> data) {setArrayInField(f,data);})
        ;
}

py::array_t<real> fieldToArray(const Field * f) {
    real* data = new real[f->datasize()];
    f->getData(data);

    // Create a Python object that will free the allocated
    // memory when destroyed
    // TODO: figure out how this works
    // https://stackoverflow.com/questions/44659924/returning-numpy-arrays-via-pybind11
    py::capsule free_when_done(data, [](void* f) {
      real* data = reinterpret_cast<real*>(f);
      delete[] data;
    });

    int shape[4];
    shape[0] = f->ncomp();
    shape[1] = f->grid().size().z;
    shape[2] = f->grid().size().y;
    shape[3] = f->grid().size().x;

    int strides[4];
    strides[0] = sizeof(real) * shape[3] * shape[2] * shape[1];
    strides[1] = sizeof(real) * shape[3] * shape[2];
    strides[2] = sizeof(real) * shape[3];
    strides[3] = sizeof(real);

    return py::array_t<real>(shape, strides, data, free_when_done);
}

void setArrayInField( Field* f, py::array_t<real> data) {
    int ndim = data.ndim();

    if (ndim == 1) {
      if (data.shape(0) != f->ncomp()){
        throw std::runtime_error("The number of components do not match");
      }

      py::buffer_info buf = data.request();
      real * cValues = (real*)buf.ptr;

      int N = f->datasize();
      int nCells = N/f->ncomp();
      real * buffer = new real[N];
      for (int i=0; i<N; i++){
        int c = i/nCells;
        buffer[i] = cValues[c];
      }

      f->setData(buffer);
      delete[] buffer;

    } else if (ndim==4) {

      int shape[ndim];
      shape[0] = f->ncomp();
      shape[1] = f->grid().size().z;
      shape[2] = f->grid().size().y;
      shape[3] = f->grid().size().x;

      for (int i = 0; i < ndim; i++) {
        if (shape[i] != data.shape(i)) {
          throw std::runtime_error(
              "The shape of the data does not match the shape of the field");
        }
      }
      py::buffer_info buf = data.request();
      f->setData((real*)buf.ptr);

    } else {
        throw std::runtime_error(
            "The shape of the data does not match the shape of the field");
    }
}