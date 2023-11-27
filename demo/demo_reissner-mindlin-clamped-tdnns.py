# ---
# jupyter:
#   jupytext:
#     text_representation:
#       extension: .py
#       format_name: light
#       format_version: '1.5'
#       jupytext_version: 1.14.1
# ---

# # Clamped Reissner-Mindlin plate under uniform load using TDNNS element
#
# This demo program solves the out-of-plane Reissner-Mindlin equations on the
# unit square with uniform transverse loading and fully clamped boundary
# conditions using the TDNNS (tangential displacement normal-normal stress)
# element developed in:
#
# Pechstein, A. S., Schöberl, J. The TDNNS method for Reissner–Mindlin plates.
# Numer. Math. 137, 713–740 (2017).
# [doi:10.1007/s00211-017-0883-9](https://doi.org/10.1007/s00211-017-0883-9)
#
# The idea behind this element construction is that the rotations, transverse
# displacement and bending moments are discretised separately. The finite
# element space for the transverse displacement is chosen such that the
# gradient is a subset of the rotation space and therefore the Kirchhoff
# constraint as the thickness parameter tends to zero is exactly satisfied by
# construction.
#
# Mathematically the forms in this work follow exactly that shown in Pechstein
# and Schöberl except for a few minor notational changes to match the rest of
# FEniCSx-Shells.
#
# It is assumed the reader understands most of the basic functionality of the
# new FEniCSx Project.
#
# We begin by importing the necessary functionality from DOLFINx, UFL and
# PETSc.

# +
import numpy as np

import dolfinx
import ufl
from dolfinx.fem import Function, FunctionSpace, dirichletbc
from dolfinx.fem.petsc import LinearProblem
from dolfinx.mesh import CellType, create_unit_square
from ufl import (FacetNormal, FiniteElement, Identity, Measure, MixedElement,
                 grad, inner, split, sym, tr)

from mpi4py import MPI

# -

# We then create a two-dimensional mesh of the mid-plane of the plate $\Omega =
# [0, 1] \times [0, 1]$.

# +
mesh = create_unit_square(MPI.COMM_WORLD, 16, 16, CellType.triangle)

# -

# The Pechstein-Schöberl element of order $k$ for the Reissner-Mindlin plate problem
# consists of:
#
# - $k$-th order vector-valued Nédélec elements of the second kind for the
#   rotations $\theta \in \mathrm{NED}_k$ and,
# - $k + 1$-th order scalar-valued Lagrange element for the transverse displacement field
#   $w \in \mathrm{CG}_{k + 1}$ and,
# - $k$-th order Hellan-Herrmann-Johnson finite elements for the bending moments, which
#   naturally discretise tensor-valued functions in $H(\mathrm{div}\;\mathrm{\bf{div}})$,
#   $M \in \mathrm{HHJ}_k$.
#
# The final element definition with $k = 3$ is:

# +
k = 3
U_el = MixedElement([FiniteElement("N2curl", ufl.triangle, k), FiniteElement("Lagrange", ufl.triangle, k + 1),
                     FiniteElement("HHJ", ufl.triangle, k)])
U = FunctionSpace(mesh, U_el)

u = ufl.TrialFunction(U)
u_t = ufl.TestFunction(U)

theta, w, M = split(u)
theta_t, w_t, M_t = split(u_t)
# -

# We assume constant material parameters; Young's modulus $E$, Poisson's ratio
# $\nu$, shear-correction factor $\kappa$, and thickness $t$.

# +
E = 10920.0
nu = 0.3
kappa = 5.0/6.0
t = 0.001
# -

# The overall weak form for the problem can be written as:
#
# Find $(\theta, w, M) \in \mathrm{NED}_k \times \mathrm{CG}_1 \times \mathrm{HHJ}_k$
# such that
#
# $$
# \left( k(M), \tilde{M} \right) + \left< \tilde{M}, \theta \right> + \left< M,
# \tilde{\theta} \right> - \left( \mu \gamma(\theta, w), \gamma(\tilde{\theta},
# \tilde{w}) \right) \\ = -\left(t^3, \tilde{w} \right) \quad \forall (\theta,
# w, M) \in \mathrm{NED}_k \times \mathrm{CG}_1 \times \mathrm{HHJ}_k,
# $$
# where $\left( \cdot, \cdot \right)$ is the usual $L^2$ inner product on the
# mesh $\Omega$, $\gamma(\theta, w) = \nabla w - \theta \in H(\mathrm{rot})$ is
# the shear-strain, $\mu = E \kappa t/(2(1 + \nu))$ the shear modulus. $k(M)$
# are the bending strains $k(\theta) = \mathrm{sym}(\nabla \theta)$ written in terms of
# the bending moments (stresses)
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
    """Discrete div-div inner product"""
    n = FacetNormal(M.ufl_domain())
    M_nn = nn(M)
    result = -inner(M, k_theta(theta))*dx + inner(M_nn("+"),
                                                  ufl.jump(theta, n))*dS + inner(M_nn, ufl.dot(theta, n))*ds
    return result


def gamma(theta, w):
    """Shear strain"""
    return grad(w) - theta


a = inner(k_M(M), M_t)*dx + inner_divdiv(M_t, theta) + inner_divdiv(M, theta_t) - \
    ((E*kappa*t)/(2.0*(1.0 + nu)))*inner(gamma(theta, w), gamma(theta_t, w_t))*dx
L = -inner(1.0*t**3, w_t)*dx

# -

# Imposition of boundary conditions requires some care. We reproduce the table
# from Pechstein and Schöberl specifying the different types of boundary condition.
#
# | Essential                            | Natural                              | Non-homogeneous term                                                |  # noqa: E501
# | ------------------------------------ | ------------------------------------ | ------------------------------------------------------------------- |  # noqa: E501
# | $w = \bar{w}$                        | $\mu(\partial_n w - \theta_n) = g_w$ | $\int_{\Gamma} g_w \tilde{w} \; \mathrm{d}s$                        |  # noqa: E501
# | $\theta_\tau = \bar{\theta}_{\tau} $ | $m_{n\tau} = g_{\theta_\tau}$        | $\int_{\Gamma} g_{\theta_\tau} \cdot \tilde{\theta} \; \mathrm{d}s$ |  # noqa: E501
# | $m_{nn} = \bar{m}_{nn}$              | $\theta_n = g_{\theta_n}$            | $\int_{\Gamma} g_{\theta_n} \tilde{m}_{nn} \; \mathrm{d}s$          |  # noqa: E501
#
# where $\theta_{n} = \theta_n$ is the normal component of the rotation,
# $\theta_{\tau} = \theta \cdot \tau $ is the tangential component of the
# rotation, $m_{n\tau} = m \cdot n - \sigma_{nn} n$ is the normal-tangential
# component of $n$, and $g_{w}$ etc. are known natural boundary data and
# $\bar{w}$ etc. are known essential boundary data.
#
# In the case of an essential boundary condition the values are enforced
# directly in the finite element space using `dolfinx.dirichletbc`. In the case
# of a homogeneous natural boundary condition the corresponding essential
# boundary condition should be dropped. In the case of a non-homogeneous
# condition an extra term must be added to the weak formulation.
#
# For a fully clamped plate we have on the entire boundary $\bar{w} = 0$
# (homogeneous essential), $\bar{\theta_\tau} = 0$ (homogeneous essential), and
# $g_{\theta_n} = 0$ (homogeneous natural).

# +


def all_boundary(x):
    return np.full(x.shape[1], True, dtype=bool)


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

theta, w, M = u_h.split()

if len(cells) > 0:
    value = w.eval(point, cells.array[0])
    print(value[0])
