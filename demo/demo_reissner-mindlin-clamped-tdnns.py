# ---
# jupyter:
#   jupytext:
#     text_representation:
#       extension: .py
#       format_name: light
#       format_version: '1.5'
#       jupytext_version: 1.14.1
# ---

# # Clamped Reissner-Mindlin plate under uniform load using TDNNS
#
# This demo program solves the out-of-plane Reissner-Mindlin equations on the
# unit square with uniform transverse loading with fully clamped boundary
# conditions using the TDNNS (tangential displacement normal-normal stress)
# approach from Pechstein and Schöberl.
#
# It is assumed the reader understands most of the basic functionality of the
# new FEniCSx Project.
#
# We begin by importing the necessary functionality from DOLFINx, UFL and
# PETSc.

import numpy as np

import dolfinx
import ufl
from dolfinx.fem import Function, FunctionSpace, dirichletbc
from dolfinx.fem.petsc import LinearProblem
from dolfinx.io.utils import XDMFFile
from dolfinx.mesh import CellType, create_unit_square
from ufl import (FacetNormal, FiniteElement, Identity, Measure, MixedElement,
                 grad, inner, split, sym, tr)

from mpi4py import MPI

# We then create a two-dimensional mesh of the mid-plane of the plate $\Omega =
# [0, 1] \times [0, 1]$. `GhostMode.shared_facet` is required as the Form will
# use Nédélec elements and DG-type restrictions.

mesh = create_unit_square(MPI.COMM_WORLD, 32, 32, CellType.triangle,
                          dolfinx.cpp.mesh.GhostMode.shared_facet)

# The Pechstein-Liberman element [1] for the Reissner-Mindlin plate problem
# consists of:
#
#
# The final element definition is

# +
U_el = MixedElement([FiniteElement("N2curl", ufl.triangle, 1), FiniteElement("Lagrange", ufl.triangle, 2),
                     FiniteElement("HHJ", ufl.triangle, 1)])
U = FunctionSpace(mesh, U_el)

u = ufl.TrialFunction(U)
u_t = ufl.TestFunction(U)

theta, w, M = split(u)
theta_t, w_t, M_t = split(u_t)
# -

# We assume constant material parameters; Young's modulus $E$, Poisson's ratio
# $\nu$, shear-correction factor $\kappa$, and thickness $t$.

E = 10920.0
nu = 0.3
kappa = 5.0/6.0
t = 0.001

dx = Measure("dx", mesh)
dS = Measure("dS", mesh)
ds = Measure("ds", mesh)


def k_theta(theta):
    """Bending strain tensor in terms of rotations"""
    return sym(grad(theta))


def k_M(M):
    """Bending strain tensor in terms of bending moments"""
    return (12.0/(E*t**3))*((1.0 + nu)*M - nu*Identity(2)*tr(M))


def nn(M):
    """Normal-normal component of tensor"""
    n = FacetNormal(M.ufl_domain())
    M_n = M*n
    M_nn = ufl.dot(M_n, n)
    return M_nn


def inner_divdiv(M, theta):
    n = FacetNormal(M.ufl_domain())
    M_nn = nn(M)
    result = -inner(M, k_theta(theta))*dx + inner(M_nn("+"), ufl.jump(theta, n))*dS + inner(M_nn, ufl.dot(theta, n))*ds
    return result


def gamma(theta, w):
    return grad(w) - theta


a = inner(k_M(M), M_t)*dx + inner_divdiv(M_t, theta) + inner_divdiv(M, theta_t) - \
    ((E*kappa*t)/(2.0*(1.0 + nu)))*inner(gamma(theta, w), gamma(theta_t, w_t))*dx
L = -inner(1.0*t**3, w_t)*dx


def all_boundary(x):
    return np.full(x.shape[1], True, dtype=bool)


def make_bc(value, V, on_boundary):
    bc = dirichletbc(value, boundary_dofs, V)
    return bc


boundary_entities = dolfinx.mesh.locate_entities_boundary(
    mesh, mesh.topology.dim - 1, all_boundary)

bcs = []
# Transverse displacement
boundary_dofs = dolfinx.fem.locate_dofs_topological(
    U.sub(1), mesh.topology.dim - 1, boundary_entities)
bcs.append(dirichletbc(np.array(0.0, dtype=np.float64), boundary_dofs, U.sub(1)))

# Fix tangential component of rotation
R = U.sub(0).collapse()[0]
boundary_dofs = dolfinx.fem.locate_dofs_topological(
    (U.sub(0), R), mesh.topology.dim - 1, boundary_entities)

theta_bc = Function(R)
bcs.append(dirichletbc(theta_bc, boundary_dofs, U.sub(0)))

problem = LinearProblem(a, L, bcs=bcs, petsc_options={
                        "ksp_type": "preonly", "pc_type": "lu", "pc_factor_mat_solver_type": "mumps"})
u_h = problem.solve()

bb_tree = dolfinx.geometry.BoundingBoxTree(mesh, 2)
point = np.array([0.5, 0.5, 0.0], dtype=np.float64)
cell_candidates = dolfinx.geometry.compute_collisions(bb_tree, point)
cells = dolfinx.cpp.geometry.compute_colliding_cells(
    mesh, cell_candidates, point)

theta, w, M = u_h.split()

if len(cells) > 0:
    value = w.eval(point, cells[0])
    print(value[0])

with XDMFFile(MPI.COMM_WORLD, "w.xdmf", "w") as f:
    f.write_mesh(mesh)
    f.write_function(w)
