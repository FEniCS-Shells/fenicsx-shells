# FEniCSx-Shells

A FEniCS Project-based library for simulating thin structures.

[![tests](https://github.com/FEniCS-Shells/fenicsx-shells/actions/workflows/tests.yml/badge.svg?branch=main)](https://github.com/FEniCS-Shells/fenicsx-shells/actions/workflows/tests.yml)
[![docs](https://img.shields.io/badge/docs-ready-success)](https://fenics-shells.github.io/fenicsx-shells)

## Description

FEniCSx-Shells is an open-source library that provides finite element-based
numerical methods for solving a wide range of thin structural models (beams,
plates and shells) expressed in the Unified Form Language (UFL) of the FEniCS
Project.

*FEniCSx-Shells is an experimental version targeting the new [FEniCSx
environment](https://github.com/fenics/dolfinx).*

The foundational aspects of the FEniCS-Shells project are described in the paper:

Simple and extensible plate and shell finite element models through automatic
code generation tools, J. S. Hale, M. Brunetti, S. P. A. Bordas, C. Maurini.
Computers & Structures, 209, 163-181,
[doi:10.1016/j.compstruc.2018.08.001](https://doi.org/10.1016/j.compstruc.2018.08.001).

## Documentation

Documentation can be viewed at https://fenics-shells.github.io/fenicsx-shells

## Features

FEniCSx-Shells currently includes implementations of the following structural models:

* Reissner-Mindlin plates.

A roadmap for future developments will be shared soon.

We are using a variety of numerical techniques for discretising the PDEs
including:

* Mixed Interpolation of Tensorial Component (MITC) reduction operators.
* Tangential Displacement Normal-Normal Derivative (TDNNS) methods.

## Citing

Please consider citing the old FEniCS-Shells paper and code if you find this
repository useful.

```
@article{hale_simple_2018,
title = {Simple and extensible plate and shell finite element models through automatic code generation tools},
volume = {209},
issn = {0045-7949},
url = {http://www.sciencedirect.com/science/article/pii/S0045794918306126},
doi = {10.1016/j.compstruc.2018.08.001},
journal = {Computers \& Structures},
author = {Hale, Jack S. and Brunetti, Matteo and Bordas, StÃ©phane P. A. and Maurini, Corrado},
month = oct,
year = {2018},
keywords = {Domain specific language, FEniCS, Finite element methods, Plates, Shells, Thin structures},
pages = {163--181},
}
```
along with the appropriate general [FEniCS citations](http://fenicsproject.org/citing).

## Authors

Jack S. Hale, University of Luxembourg, Luxembourg.

FEniCS-Shellsx contains code from the original FEniCS-Shells project
hosted at https://bitbucket.org/unilucompmech/fenics-shells.

## Contributing

We are always looking for contributions and help with FEniCSx-Shells. If you
have ideas, nice applications or code contributions then we would be happy to
help you get them included. We ask you to follow the FEniCS Project git
workflow.

## Issues and Support

Please use the GitHub issue tracker to report any issues.

## License

FEniCS-Shellsx is free software: you can redistribute it and/or
modify it under the terms of the GNU Lesser General Public License as published
by the Free Software Foundation, either version 3 of the License, or (at your
option) any later version.

This program is distributed in the hope that it will be useful, but WITHOUT ANY
WARRANTY; without even the implied warranty of MERCHANTABILITY or FITNESS FOR A
PARTICULAR PURPOSE.  See the GNU Lesser General Public License for more
details.

You should have received a copy of the GNU Lesser General Public License along
with fenics-shells.  If not, see http://www.gnu.org/licenses/.
