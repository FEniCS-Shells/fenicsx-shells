# ---
# jupyter:
#   jupytext:
#     text_representation:
#       extension: .py
#       format_name: light
#       format_version: '1.5'
#       jupytext_version: 1.14.1
# ---

# # Clamped Kirchoff-Love plate under uniform load
#
# This demo program solves the out-of-plane Kirchoff-Love equations on the
# unit square with uniform transverse loading and fully clamped boundary
# conditions.
#
# It is assumed the reader understands most of the basic functionality of the
# new FEniCSx Project.
#
# This demo illustrates how to:
#
# - Define the Kirchhoff-Love plate equations using UFL using the mixed finite
#   element formulation of Hellan-Herrmann-Johnson.
#
# A modern presentation of this approach can be found in the paper
#
# Arnold, D. N., Walker S. W., The Hellan--Herrmann--Johnson Method with Curved
# Elements, SIAM Journal on Numerical Analysis 58:5, 2829-2855 (2020),
# [doi:10.1137/19M1288723](https://doi.org/10.1137/19M1288723).
#
# We remark that this model can be recovered formally from the Reissner-Mindlin
# models by taking the limit in the thickness $t \to 0$ and setting $\theta =
# \grad w$.
#
# We begin by importing the necessary functionality from DOLFINx, UFL and
# PETSc.

import numpy as np

import dolfinx
import ufl
from dolfinx.fem import FunctionSpace, dirichletbc
from dolfinx.fem.petsc import LinearProblem
from dolfinx.mesh import CellType, create_unit_square
from ufl import (FacetNormal, FiniteElement, Identity, Measure, MixedElement,
                 grad, inner, sym, tr)

from mpi4py import MPI

# We then create a two-dimensional mesh of the mid-plane of the plate $\Omega =
# [0, 1] \times [0, 1]$.

mesh = create_unit_square(MPI.COMM_WORLD, 16, 16, CellType.triangle)

# The Hellen-Herrmann-Johnson element for the Kirchhoff-Love plate problem
# consists of:
#
# - $k + 1$-th order scalar-valued Lagrange element for the transverse displacement field
#   $w \in \mathrm{CG}_{k + 1}$ and,
# - $k$-th order Hellan-Herrmann-Johnson finite elements for the bending moments, which
#   naturally discretise tensor-valued functions in $H(\mathrm{div}\;\mathrm{\bf{div}})$,
#   $M \in \mathrm{HHJ}_k$.
#
# The final element definition is

# +
k = 2
U_el = MixedElement([FiniteElement("Lagrange", ufl.triangle, k + 1),
                     FiniteElement("HHJ", ufl.triangle, k)])
U = FunctionSpace(mesh, U_el)

w, M = ufl.TrialFunctions(U)
w_t, M_t = ufl.TestFunctions(U)

# -

# We assume constant material parameters; Young's modulus $E$, Poisson's ratio
# $\nu$, shear-correction factor $\kappa$, and thickness $t$.

# +
E = 10920.0
nu = 0.3
t = 0.001

# -

# The weak form for the problem can be written as:
#
# Find $(w, M) \in \mathrm{CG}_{k + 1} \times \mathrm{HHJ}_k$ such that
#
# $$
# \left( k(M), \tilde{M} \right) + \left< \tilde{M}, \theta(w) \right> + \left<
# M, \theta(\tilde{w}) \right> = -\left(f, \tilde{w} \right) \quad \forall
# (\tilde{w}, \tilde{M}) \in \mathrm{CG}_{k + 1} \times \mathrm{HHJ}_k,
# $$
# where $\left( \cdot, \cdot \right)$ is the usual $L^2$ inner product on the
# mesh $\Omega$.
# The rotations $\theta$ for the Kirchhoff-Love model can be written in
# terms of the transverse displacements
#
# $$
# \theta(w) = \nabla w.
# $$
#
# The bending strain tensor $k$ for the Kirchoff-Love model can be expressed
# in terms of the rotations
#
# $$
# k(\theta) = \dfrac{1}{2}(\nabla \theta + (\nabla \theta)^T).
# $$
#
# The bending strain tensor $k$ can also be written in terms of the bending
# moments $M$
#
# $$
# k(M) = \frac{12}{E t^3} \left[ (1 + \nu) M - \nu \mathrm{tr}\left( M \right) I \right],
# $$
# with $\mathrm{tr}$ the trace operator and $I$ the identity tensor.
#
# The inner product $\left< \cdot, \cdot \right>$ is defined by
#
# $$
# \left< M, \theta \right> = -\left( M, k(\theta) \right) + \int_{\partial K}
# M_{nn} \cdot [[ \theta ]]_n \; \mathrm{d}s,
# $$
# where $M_{nn} = \left(Mn \right) \cdot n$ is the normal-normal component of
# the bending moment, $\partial K$ are the facets of the mesh, $[[ \theta ]]$ is
# the jump in the normal component of the rotations on the facets (reducing to
# simply $\theta \cdot n$ on the exterior facets).
#
# The above equations can be written relatively straightforwardly in UFL as:

# +
dx = Measure("dx", mesh)
dS = Measure("dS", mesh)
ds = Measure("ds", mesh)


def theta(w):
    """Rotations in terms of transverse displacements"""
    return grad(w)


def k_theta(theta):
    """Bending strain tensor in terms of rotations"""
    return sym(grad(theta))


def k_M(M):
    """Bending strain tensor in terms of bending moments"""
    return (12.0/(E*t**3))*((1.0 + nu)*M - nu*Identity(2)*tr(M))


def nn(M):
    """Normal-normal component of tensor"""
    n = FacetNormal(M.ufl_domain())
    M_n = ufl.dot(M, n)
    M_nn = ufl.dot(M_n, n)
    return M_nn


def inner_divdiv(M, theta):
    """Discrete div-div inner product"""
    n = FacetNormal(M.ufl_domain())
    M_nn = nn(M)
    result = -inner(M, k_theta(theta))*dx + inner(M_nn("+"),
                                                  ufl.jump(theta, n))*dS + inner(M_nn, ufl.dot(theta, n))*ds
    return result


a = inner(k_M(M), M_t)*dx + inner_divdiv(M_t, theta(w)) + \
    inner_divdiv(M, theta(w_t))
L = -inner(t**3, w_t)*dx


def all_boundary(x):
    return np.full(x.shape[1], True, dtype=bool)


# -

# We apply clamped boundary conditions on the entire boundary. The essential
# boundary condition $w = 0$ is enforced directly in the finite element space,
# while the condition $\nabla w \cdot n = 0$ is a natural condition that is
# satisfied when the corresponding essential condition on $m_{nn}$ is dropped.

# TODO: Add table like TDNNS example.

# +
boundary_entities = dolfinx.mesh.locate_entities_boundary(
    mesh, mesh.topology.dim - 1, all_boundary)

bcs = []
# Transverse displacement
boundary_dofs_displacement = dolfinx.fem.locate_dofs_topological(
    U.sub(0), mesh.topology.dim - 1, boundary_entities)
bcs.append(dirichletbc(np.array(0.0, dtype=np.float64),
           boundary_dofs_displacement, U.sub(0)))


# -

# Finally we solve the problem and output the transverse displacement at the
# centre of the plate.

# +
problem = LinearProblem(a, L, bcs=bcs, petsc_options={
                        "ksp_type": "preonly", "pc_type": "lu", "pc_factor_mat_solver_type": "mumps"})
u_h = problem.solve()

bb_tree = dolfinx.geometry.bb_tree(mesh, 2)
point = np.array([[0.5, 0.5, 0.0]], dtype=np.float64)
cell_candidates = dolfinx.geometry.compute_collisions_points(bb_tree, point)
cells = dolfinx.geometry.compute_colliding_cells(
    mesh, cell_candidates, point)

w, M = u_h.split()

if len(cells) > 0:
    value = w.eval(point, cells.array[0])
    print(value[0])

# TODO: IO?
