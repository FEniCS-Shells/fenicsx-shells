cmake_minimum_required(VERSION 3.16)

project(fenics_shellsx VERSION "0.1.0")

set(CMAKE_MODULE_PATH ${CMAKE_MODULE_PATH} ${CMAKE_CURRENT_SOURCE_DIR}/fenics_shellsx/cpp/cmake)

set(Python_FIND_IMPLEMENTATIONS CPython)
find_package(Python REQUIRED COMPONENTS Interpreter Development)
find_package(Eigen3 REQUIRED)
find_package(DOLFINX REQUIRED)
find_package(PETSc4py REQUIRED)

# Set C++ standard before finding pybind11
set(CMAKE_CXX_STANDARD 17)
set(CMAKE_CXX_STANDARD_REQUIRED ON)
set(CMAKE_CXX_EXTENSIONS OFF)

# Now we can find pybind11
find_package(pybind11 CONFIG REQUIRED)

pybind11_add_module(_fenics_shellsxcpp MODULE fenics_shellsx/cpp/wrapper.cpp)

target_sources(_fenics_shellsxcpp PRIVATE fenics_shellsx/cpp/assembler.cpp)

# Add DOLFINx libraries and other config
target_include_directories(_fenics_shellsxcpp PRIVATE ${PETSC4PY_INCLUDE_DIRS})
target_link_libraries(_fenics_shellsxcpp PRIVATE pybind11::module dolfinx Eigen3::Eigen)
