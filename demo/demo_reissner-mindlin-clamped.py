# ---
# jupyter:
#   jupytext:
#     text_representation:
#       extension: .py
#       format_name: light
#       format_version: '1.5'
#       jupytext_version: 1.14.1
# ---

# # Clamped Reissner-Mindlin plate under uniform load
#
# This demo program solves the out-of-plane Reissner-Mindlin equations on the
# unit square with uniform transverse loading with fully clamped boundary
# conditions. This version does not use the special projected assembly routines
# in FEniCSx-Shells.
#
# It is assumed the reader understands most of the basic functionality of the
# new FEniCSx Project.
#
# This demo illustrates how to:
#
# - Define the Reissner-Mindlin plate equations using UFL.
# - Define the Durán-Liberman (MITC) reduction operator using UFL. This
#   procedure eliminates the shear-locking problem.
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
from ufl import (FiniteElement, MixedElement, VectorElement, dx, grad, inner,
                 split, sym, tr)

from mpi4py import MPI

# We then create a two-dimensional mesh of the mid-plane of the plate $\Omega =
# [0, 1] \times [0, 1]$. `GhostMode.shared_facet` is required as the Form will
# use Nédélec elements and DG-type restrictions.

mesh = create_unit_square(MPI.COMM_WORLD, 32, 32, CellType.triangle)

# The Durán-Liberman element [1] for the Reissner-Mindlin plate problem
# consists of:
#
# - second-order vector-valued Lagrange element for the rotation field $\theta
#   \in [\mathrm{CG}_2]^2$ and,
# - a first-order scalar valued Lagrange element for the transverse
#   displacement field $w \in \mathrm{CG}_1$ and,
# - the reduced shear strain $\gamma_R \in \mathrm{NED}_1$ the vector-valued
#   Nédélec elements of the first kind, and
# - a Lagrange multiplier field $p$ that ties together the shear strain
#   calculated from the primal variables $\gamma = \nabla w - \theta$ and the
#   reduced shear strain $\gamma_R$. Both $p$ and $\gamma_R$ are are discretised
#   in the space $\mathrm{NED}_1$, the vector-valued Nédélec elements of the
#   first kind.
#
# The final element definition is

# +
U_el = MixedElement([VectorElement("Lagrange", ufl.triangle, 2), FiniteElement("Lagrange", ufl.triangle, 1),
                     FiniteElement("N1curl", ufl.triangle, 1), FiniteElement("N1curl", ufl.triangle, 1)])
U = FunctionSpace(mesh, U_el)

u_ = Function(U)
u = ufl.TrialFunction(U)
u_t = ufl.TestFunction(U)

theta_, w_, R_gamma_, p_ = split(u_)
# -

# We assume constant material parameters; Young's modulus $E$, Poisson's ratio
# $\nu$, shear-correction factor $\kappa$, and thickness $t$.

E = 10920.0
nu = 0.3
kappa = 5.0/6.0
t = 0.001

# The bending strain tensor $k$ for the Reissner-Mindlin model can be expressed
# in terms of the rotation field $\theta$
#
# $$
# k(\theta) = \dfrac{1}{2}(\nabla \theta + (\nabla \theta)^T)
# $$
#
# The bending energy density $\psi_b$ for the Reissner-Mindlin model is a
# function of the bending strain tensor $k$
#
# $$
# \psi_b(k) = \frac{1}{2} D \left( (1 - \nu) \, \mathrm{tr}\,(k^2) + \nu \, (\mathrm{tr}    \,k)^2 \right) \qquad
# D = \frac{Et^3}{12(1 - \nu^2)}
# $$
#
# which can be expressed in UFL as

D = (E*t**3)/(24.0*(1.0 - nu**2))
k = sym(grad(theta_))
psi_b = D*((1.0 - nu)*tr(k*k) + nu*(tr(k))**2)

# Because we are using a mixed variational formulation, we choose to write the
# shear energy density $\psi_s$ is a function of the reduced shear strain
# vector
#
# $$\psi_s(\gamma_R) = \frac{E \kappa t}{4(1 + \nu)}\gamma_R^2,$$
#
# or in UFL:

psi_s = ((E*kappa*t)/(4.0*(1.0 + nu)))*inner(R_gamma_, R_gamma_)

# Finally, we can write out external work due to the uniform loading in the out-of-plane direction
#
# $$
# W_{\mathrm{ext}} = \int_{\Omega} ft^3 \cdot w \; \mathrm{d}x.
# $$

# where $f = 1$ and $\mathrm{d}x$ is a measure on the whole domain.
# The scaling by $t^3$ is included to ensure a correct limit solution as
# $t \to 0$.
#
# In UFL this can be expressed as

W_ext = inner(1.0*t**3, w_)*dx

# With all of the standard mechanical terms defined, we can turn to defining
# the Duran-Liberman reduction operator. This operator 'ties' our reduced shear
# strain field to the shear strain calculated in the primal space.  A partial
# explanation of the thinking behind this approach is given in the Appendix at
# the bottom of this notebook.
#
# The shear strain vector $\gamma$ can be expressed in terms of the rotation
# and transverse displacement field
#
# $$\gamma(\theta, w) = \nabla w - \theta$$
#
# or in UFL

gamma = grad(w_) - theta_

# We require that the shear strain calculated using the displacement unknowns
# $\gamma = \nabla w - \theta$ be equal, in a weak sense, to the conforming
# shear strain field $\gamma_R \in \mathrm{NED}_1$ that we used to define the
# shear energy above.  We enforce this constraint using a Lagrange multiplier
# field $p \in \mathrm{NED}_1$. We can write the Lagrangian functional of this
# constraint as:
#
# $$\Pi_R(\gamma, \gamma_R, p) =
#   \int_{e} \left( \left\lbrace \gamma_R - \gamma \right\rbrace \cdot t \right)
#   \cdot \left( p \cdot t \right) \; \mathrm{d}s$$
#
# where $e$ are all of edges of the cells in the mesh and $t$ is the tangent
# vector on each edge.
#
# Writing this operator out in UFL is quite verbose, so `fenicsx_shells`
# includes a special inner product function `inner_e` to help. However, we
# choose to write this function in full here.

# +
dSp = ufl.Measure('dS', metadata={'quadrature_degree': 1})
dsp = ufl.Measure('ds', metadata={'quadrature_degree': 1})

n = ufl.FacetNormal(mesh)
t = ufl.as_vector((-n[1], n[0]))


def inner_e(x, y):
    return (inner(x, t)*inner(y, t))('+') * \
        dSp + (inner(x, t)*inner(y, t))*dsp


Pi_R = inner_e(gamma - R_gamma_, p_)
# -

# We can now define our Lagrangian for the complete system and derive the
# residual and Jacobian automatically using the standard UFL `derivative`
# function

Pi = psi_b*dx + psi_s*dx + Pi_R - W_ext
F = ufl.derivative(Pi, u_, u_t)
J = ufl.derivative(F, u_, u)

# In the following we use standard from `dolfinx` to apply boundary conditions,
# assemble, solve and output the solution.
#
# Note that for a fully clamped plate $\gamma_R \in H_0(\mathrm{rot}; \Omega)$
# which is the Sobolev space of vector-valued functions with square integrable
# whose tangential component $\gamma_R \cdot t$ vanishes on the boundary. So,
# it is also necessary to apply boundary conditions on this space.
#
# For simplicity of implementation we also apply boundary conditions on the
# Lagrange multiplier space but this is not strictly necessary as the Lagrange
# multiplier simply constrains $\gamma_R \cdot t$ to $\gamma = \nabla w -
# \theta$, all of which are enforced to be zero by definition.

# +


def all_boundary(x):
    return np.full(x.shape[1], True, dtype=bool)


u_boundary = Function(U)
boundary_entities = dolfinx.mesh.locate_entities_boundary(
    mesh, mesh.topology.dim - 1, all_boundary)
boundary_dofs = dolfinx.fem.locate_dofs_topological(
    U, mesh.topology.dim - 1, boundary_entities)
bcs = [dirichletbc(u_boundary, boundary_dofs)]

problem = LinearProblem(J, -F, bcs=bcs, petsc_options={
                        "ksp_type": "preonly", "pc_type": "lu", "pc_factor_mat_solver_type": "mumps"})
u_ = problem.solve()

bb_tree = dolfinx.geometry.bb_tree(mesh, 2)
point = np.array([[0.5, 0.5, 0.0]], dtype=np.float64)
cell_candidates = dolfinx.geometry.compute_collisions_points(bb_tree, point)
cells = dolfinx.geometry.compute_colliding_cells(
    mesh, cell_candidates, point)

theta, w, R_gamma, p = u_.split()

if len(cells) > 0:
    value = w.eval(point, cells.array[0])
    print(value[0])
    # NOTE: FEniCS-Shells (old dolfin) `demo/documented/reissner-mindlin-clamped`
    # gives 1.28506469462e-06 on a 32 x 32 mesh and 1.2703580973e-06 on a 64 x 64
    # mesh.

    def test_center_displacement():
        assert np.isclose(value[0], 1.285E-6, atol=1E-3, rtol=1E-3)

with XDMFFile(MPI.COMM_WORLD, "w.xdmf", "w") as f:
    f.write_mesh(mesh)
    f.write_function(w)

# -

# ## Appendix

# For the clamped problem we have the following regularity for our two fields,
# $\theta \in [H^1_0(\Omega)]^2$ and $w \in [H^1_0(\Omega)]^2$ where
# $H^1_0(\Omega)$ is the usual Sobolev space of functions with square
# integrable first derivatives that vanish on the boundary. If we then take
# $\nabla w$ we have the result $\nabla w \in H_0(\mathrm{rot}; \Omega)$ which
# is the Sobolev space of vector-valued functions with square integrable
# $\mathrm{rot}$ whose tangential component $\nabla w \cdot t$ vanishes on the
# boundary.
#
# Let's look at our expression for the shear strain vector in light of these
# new results. In the thin-plate limit $t \to 0$, we would like to recover our
# the standard Kirchhoff-Love problem where we do not have transverse shear
# strains $\gamma \to 0$ at all. In a finite element context, where we have
# discretised fields $w_h$ and $\theta_h$ we then would like
#
# $$\gamma(\theta_h, w_h) := \nabla w_h - \theta_h = 0 \quad t \to 0 \; \forall x \in \Omega$$
#
# If we think about using first-order piecewise linear polynomial finite
# elements for both fields, then we are requiring that piecewise constant
# functions ($\nabla w_h$) are equal to piecewise linear functions ($\theta_h$)
# # ! This is strong requirement, and is the root of the famous shear-locking
# problem. The trick of the Durán-Liberman approach is recognising that by
# modifying the rotation field at the discrete level by applying a special
# operator $R_h$ that takes the rotations to the conforming space
# $\mathrm{NED}_1 \subset H_0(\mathrm{rot}; \Omega)$ for the shear strains that
# we previously identified:
#
# $$R_h : H_0^1(\Omega) \to H_0(\mathrm{rot}; \Omega)$$
#
# we can 'unlock' the element. With this reduction operator applied as follows:
#
# $$\gamma(\theta_h, w_h) := R_h(\nabla w_h - \theta_h = 0) \quad t \to 0 \; \forall x \in \Omega$$
#
# our requirement of vanishing shear strains can actually hold. This is the
# basic mathematical idea behind all MITC approaches, of which the
# Durán-Liberman approach is a specific implementation.
