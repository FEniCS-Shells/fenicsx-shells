#!/usr/bin/env python
import numpy as np

from mpi4py import MPI
from petsc4py import PETSc

import dolfinx
from dolfinx import UnitSquareMesh, FunctionSpace
from dolfinx.cpp.mesh import CellType
from dolfinx.fem import assemble_matrix

import ufl
from ufl import FiniteElement, MixedElement
from ufl import dx, inner

mesh = UnitSquareMesh(MPI.COMM_WORLD, 1, 1, CellType.triangle,
                      dolfinx.cpp.mesh.GhostMode.shared_facet)

#U_el = MixedElement([FiniteElement("N1curl", ufl.triangle, 1)])
U_el = FiniteElement("N1curl", ufl.triangle, 1)
U = FunctionSpace(mesh, U_el)

u = ufl.TrialFunction(U)
v = ufl.TestFunction(U)

print(u)

a = inner(u, v)*dx

A = dolfinx.fem.assemble_matrix(a)
A.assemble()
print(A.convert('dense').getDenseArray())
print(A.norm())
