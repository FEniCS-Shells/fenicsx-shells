#include <pybind11/pybind11.h>

namespace py = pybind11;

PYBIND11_MODULE(_fenicsx_shellscpp, m)
{
  m.attr("__version__") = "0.1.0.0";

  m.def("test", []() { return 0; });
}
