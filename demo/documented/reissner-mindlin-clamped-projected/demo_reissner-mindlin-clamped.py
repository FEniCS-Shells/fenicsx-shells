#!/usr/bin/env python
# coding: utf-8

# # Clamped Reissner-Mindlin plate under uniform load
# ## Pure DOLFINX implementation
#
# This demo program solves the out-of-plane Reissner-Mindlin equations on the
# unit square with uniform transverse loading with fully clamped boundary
# conditions. This version does not use the special projected assembly routines in FEniCS-ShellsX.
#
# It is assumed the reader understands most of the basic functionality
# of the new FEniCSX Project.
#
# This demo illustrates how to:
#
# - Define the Reissner-Mindlin plate equations using UFL.
# - Define the Durán-Liberman (MITC) reduction operator using UFL. This procedure
#   eliminates the shear-locking problem.
#
# We begin by importing the necessary functionality from DOLFINX, UFL and PETSc.

# In[1]:


import numpy as np

from mpi4py import MPI
from petsc4py import PETSc

import dolfinx
from dolfinx import UnitSquareMesh, Function, FunctionSpace, DirichletBC, Constant
from dolfinx.cpp.mesh import CellType
from dolfinx.fem import (apply_lifting, assemble_matrix, assemble_vector,
                         locate_dofs_topological, set_bc)
from dolfinx.mesh import locate_entities_boundary
from dolfinx.io import XDMFFile

import ufl
from ufl import FiniteElement, MixedElement, VectorElement
from ufl import sym, grad, tr, dx, inner, split

from fenics_shellsx import assemble

mesh = UnitSquareMesh(MPI.COMM_WORLD, 32, 32, CellType.triangle,
                      dolfinx.cpp.mesh.GhostMode.shared_facet)

U_el = MixedElement([MixedElement([VectorElement("Lagrange", ufl.triangle, 2), FiniteElement("Lagrange", ufl.triangle, 1)]),
                     MixedElement([FiniteElement("N1curl", ufl.triangle, 1), FiniteElement("N1curl", ufl.triangle, 1)])])
U = FunctionSpace(mesh, U_el)

u_ = Function(U)
u = ufl.TrialFunction(U)
u_t = ufl.TestFunction(U)

primal_, mixed_ = split(u_)
theta_, w_ = split(primal_)
R_gamma_, p_ = split(mixed_)

E = 10920.0
nu = 0.3
kappa = 5.0/6.0
t = 0.001

D = (E*t**3)/(24.0*(1.0 - nu**2))
k = sym(grad(theta_))
psi_b = D*((1.0 - nu)*tr(k*k) + nu*(tr(k))**2)

psi_s = ((E*kappa*t)/(4.0*(1.0 + nu)))*inner(R_gamma_, R_gamma_)

W_ext = inner(1.0*t**3, w_)*dx

gamma = grad(w_) - theta_

dSp = ufl.Measure('dS', metadata={'quadrature_degree': 1})
dsp = ufl.Measure('ds', metadata={'quadrature_degree': 1})

n = ufl.FacetNormal(mesh)
t = ufl.as_vector((-n[1], n[0]))

def inner_e(x, y): return (inner(x, t)*inner(y, t))('+')*dSp + \
    (inner(x, t)*inner(y, t))('-')*dSp + (inner(x, t)*inner(y, t))*dsp


Pi_R = inner_e(gamma - R_gamma_, p_)

Pi = psi_b*dx + psi_s*dx + Pi_R - W_ext
F = ufl.derivative(Pi, u_, u_t)
J = ufl.derivative(F, u_, u)

u0 = Function(U)
u0.vector.set(0.0)
facets = locate_entities_boundary(
    mesh, 1, lambda x: np.ones(x.shape[1], dtype=bool))
dofs0 = locate_dofs_topological(U, 1, facets)
bcs = [DirichletBC(u0, dofs0)]

J_dolfin = dolfinx.fem.Form(J)
F_dolfin = dolfinx.fem.Form(F)
A, b = assemble(J_dolfin._cpp_object, F_dolfin._cpp_object)
A.assemble()

ksp = PETSc.KSP().create(MPI.COMM_WORLD)

pc = ksp.getPC()
pc.setType("lu")
pc.setFactorSolverType("mumps")

ksp.setOperators(A)
ksp.setType("preonly")
ksp.setFromOptions()
ksp.solve(b, u_.vector)

bb_tree = dolfinx.cpp.geometry.BoundingBoxTree(mesh, 2)
point = np.array([0.5, 0.5, 0.0], dtype=np.float64)
cell_candidates = dolfinx.cpp.geometry.compute_collisions_point(bb_tree, point)
cell = dolfinx.cpp.geometry.select_colliding_cells(
    mesh, cell_candidates, point, 1)

primal, dual = u_.split()
w, theta = primal.split()
R_gamma, p = dual.split()

if len(cell) > 0:
    value = w.eval(point, cell)
    print(value[0])
    # NOTE: FEniCS-Shells (old dolfin) `demo/documented/reissner-mindlin-clamped`
    # gives 1.28506469462e-06 on a 32 x 32 mesh and 1.2703580973e-06 on a 64 x 64
    # mesh.

    def test_center_displacement():
        assert(np.isclose(value[0], 1.285E-6, atol=1E-3, rtol=1E-3))

with XDMFFile(MPI.COMM_WORLD, "w.xdmf", "w") as f:
    f.write_mesh(mesh)
    f.write_function(w)

with XDMFFile(MPI.COMM_WORLD, "theta.xdmf", "w") as f:
    f.write_mesh(mesh)
    f.write_function(theta)
