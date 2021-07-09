#include <pybind11/pybind11.h>

#include "assembler.h"

namespace py = pybind11;

PYBIND11_MODULE(_fenics_shellsxcpp, m)
{
  m.attr("__version__") = "0.1.0";

  m.def("test", []() { return 0; });
  m.def("assemble", &assemble<double>);
}
