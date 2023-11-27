cmake_minimum_required(VERSION 3.18)

project(fenicsx_shells VERSION "0.1.0.0")

find_package(Python COMPONENTS Interpreter Development REQUIRED)
find_package(DOLFINX REQUIRED)

# Set C++ standard before finding pybind11
set(CMAKE_CXX_STANDARD 20)
set(CMAKE_CXX_STANDARD_REQUIRED ON)
set(CMAKE_CXX_EXTENSIONS OFF)

# Now we can find pybind11
find_package(pybind11 CONFIG REQUIRED)

pybind11_add_module(_fenicsx_shellscpp MODULE fenicsx_shells/cpp/wrapper.cpp)

target_link_libraries(_fenicsx_shellscpp PRIVATE pybind11::module)
