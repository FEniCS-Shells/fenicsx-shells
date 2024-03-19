# %% [markdown]
# # Clamped semi-cylindrical Naghdi shell under point load
#
# Modified by: Tian Yang
#
# %% [markdown]
# This demo program solves the nonlinear Naghdi shell equations for a semi-cylindrical shell loaded by a point force.
# This problem is a standard reference for testing shell finite element formulations, see [1]. The numerical locking
# issue is cured using enriched finite element including cubic bubble shape functions and Partial Selective Reduced
# Integration [2].
#
# It is assumed the reader understands most of the basic functionality of the new FEniCSx Project.
#
# This demo then illustrates how to:
#
# - Define and solve a nonlinear Naghdi shell problem with a curved stress-free configuration given as analytical
# expression in terms of two curvilinear coordinates.
# - Use the PSRI approach to simultaneously cure shear- and membrane-locking issues.
#
# We begin by importing the necessary functionality from DOLFINx, UFL and PETSc.
#
# %%
import typing
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
# %%
import dolfinx
import ufl
from dolfinx.fem import (Expression, Function, FunctionSpace, dirichletbc,
                         locate_dofs_topological)
from dolfinx.fem.bcs import DirichletBC
from dolfinx.fem.function import Function as _Function
from dolfinx.fem.petsc import (NonlinearProblem, apply_lifting,
                               assemble_vector, set_bc)
from dolfinx.mesh import CellType, create_rectangle, locate_entities_boundary
from dolfinx.nls.petsc import NewtonSolver
from ufl import FiniteElement, MixedElement, VectorElement, grad, inner, split
# %%
from mpi4py import MPI
from petsc4py import PETSc
# %% [markdown]
# We consider a semi-cylindrical shell of radius $r$ and axis length $L$. The shell is made of a linear elastic
# isotropic homogeneous
# material with Young modulus $E$ and Poisson ratio $\nu$. The
# (uniform) shell thickness is denoted by $t$.
# The Lamé moduli $\lambda$, $\mu$ are introduced to write later
# the 2D constitutive equation in plane-stress:
#
# %%
r = 1.016
L = 3.048
E, nu = 2.0685E7, 0.3
mu = E/(2.0*(1.0 + nu))
lmbda = 2.0*mu*nu/(1.0 - 2.0*nu)
t = 0.03
#
# %% [markdown]
# The midplane of the initial (stress-free) configuration $\vec{\phi_0}$ of the shell is given in the form of an
# analytical expression:
#
# $$
# \vec{\phi}_0(\xi_1, \xi_2) \subset \mathbb{R}³
# $$
#
# where $\xi_1 \in [-\pi/2, \pi/2]$ and $\xi_2 \in [0, L]$ are the curvilinear coordinates. In this case, they
# represent the angular and axial coordinates, respectively.
#
# %% [markdown]
# We generate a mesh in the $(\xi_1, \xi_2)$ space with triangular elements
#
# %%
mesh = create_rectangle(MPI.COMM_WORLD, np.array([[-np.pi / 2, 0], [np.pi / 2, L]]), [20, 20], CellType.triangle)
# topology dimension = 2
tdim = mesh.topology.dim

# %% [markdown]
# We provide the analytical expression of the initial shape as a `ufl` expression
#
# %%
x = ufl.SpatialCoordinate(mesh)
phi0_ufl = ufl.as_vector([r * ufl.sin(x[0]), x[1], r * ufl.cos(x[0])])
# %% [markdown]
# Given the analytical expression of midplane, we define the unit normal as below:
#
# $$
# \vec{n}  = \frac{\partial_1 \phi_0 \times \partial_2 \phi_0}{\| \partial_1 \phi_0 \times \partial_2 \phi_0 \|}
# $$
#
# %%


def unit_normal(phi):
    n = ufl.cross(phi.dx(0), phi.dx(1))
    return n/ufl.sqrt(inner(n, n))


n0_ufl = unit_normal(phi0_ufl)
# %% [markdown]
# We define a local orthonormal frame $\{\vec{t}_{01}, \vec{t}_{02}, \vec{n}\}$ of the initial configuration $\phi_0$
# by rotating the global Cartesian basis $\vec{e}_i$ with a rotation matrix $\mathbf{R}_0$:
#
# $$
# \vec{t}_{0i} = \mathbf{R}_0 \vec{e}_i , \quad \vec{n} = \vec{t}_{03},
# $$
#
# A convienent choice of $\vec{t}_{01}$ and $\vec{t}_{02}$ (when $\vec{n} \nparallel \vec{e}_2 $) could be:
# $$
# \vec{t}_{01} = \frac{\vec{e}_2 \times \vec{n}}{\| \vec{e}_2 \times \vec{n}\|} \\
# \vec{t}_{02} =   \vec{n} \times \vec{t}_{01}
# $
#
#
# The corresponding rotation matrix $\mathbf{R}_0$:
# $$
# \mathbf{R}_0 = [\vec{t}_{01}; \vec{t}_{02}; \vec{n}]
# $
#
# %%


def tangent_1(n):
    e2 = ufl.as_vector([0, 1, 0])
    t1 = ufl.cross(e2, n)
    t1 = t1/ufl.sqrt(inner(t1, t1))
    return t1


def tangent_2(n, t1):
    t2 = ufl.cross(n, t1)
    t2 = t2/ufl.sqrt(inner(t2, t2))
    return t2


# the analytical expression of t1 and t2
t1_ufl = tangent_1(n0_ufl)
t2_ufl = tangent_2(n0_ufl, t1_ufl)


# the analytical expression of R0
def rotation_matrix(t1, t2, n):
    R = ufl.as_matrix([[t1[0], t2[0], n[0]],
                       [t1[1], t2[1], n[1]],
                       [t1[2], t2[2], n[2]]])
    return R


R0_ufl = rotation_matrix(t1_ufl, t2_ufl, n0_ufl)
# %% [markdown]
# The kinematics of the Nadghi shell model is defined by the following vector fields :
# - $\vec{\phi}$: the position of the midplane in the deformed configuration, or equivalently, the displacement
# $\vec{u} = \vec{\phi} - \vec{\phi}_0$
# - $\vec{d}$: the director, a unit vector giving the orientation of fiber at the midplane. (not necessarily normal to
# the midsplane because of shears)
#
# %% [markdown]
# According to [3], the director $\vec{d}$ in the deformed configuration can be parameterized with two successive
# rotation angles $\theta_1, \theta_2$
#
# $$
# \vec{t}_i = \mathbf{R} \vec{e}_i, \quad \mathbf{R}  = \text{exp}[\theta_1 \hat{\mathbf{t}}_1] \text{exp}[\theta_2
# \hat{\mathbf{t}}_{02}] \mathbf{R}_0
# $$
#
# The rotation matrix $\mathbf{R}$ represents three successive rotations:
# - First one: the initial rotation matrix $\mathbf{R}_0$
# - Second one :$\text{exp}[\theta_2 \hat{\mathbf{t}}_{02}]$ rotates a vector about the axis $\vec{t}_{02}$ of
# $\theta_2$ angle;
# - Third one : $\text{exp}[\theta_1 \hat{\mathbf{t}}_1]$ rotates a vector about the axis $\vec{t}_{1}$ of $\theta_1$
# angle, and $\vec{t}_1 = \text{exp}[\theta_2 \hat{\mathbf{t}}_{02}] \vec{t}_{01}$
#
# The rotation matrix $\mathbf{R}$ on the other hand is equivalent to rotate around the fixed axis $\vec{e}_1$ and
# $\vec{e}_2$ (Proof see [3]):
#
# $$
# \mathbf{R} = \mathbf{R}_0 \text{exp}[\theta_2 \hat{\mathbf{e}}_{2}] \text{exp}[\theta_1 \hat{\mathbf{e}}_1]
# $$
#
# Therefore, the director $\vec{d}$ is updated with $(\theta_1, \theta_2)$ by:
#
# $$
# \vec{d} =\mathbf{R} \vec{e}_3 = \mathbf{R}_0 \vec{\Lambda}_3, \quad \vec{\Lambda}_3 =
# [\sin(\theta_2)\cos(\theta_1), -\sin(\theta_1), \cos(\theta_2)\cos(\theta_1)]^\text{T}
# $$
#
# Note: the above formular becomes singular when $\theta_1 = \pm \pi/2, ...$, (See Chapter 4.2.1 in [3] for details)
# %%
# Update the director with two successive elementary rotations


def director(R0, theta):
    Lm3 = ufl.as_vector([ufl.sin(theta[1])*ufl.cos(theta[0]), -ufl.sin(theta[0]), ufl.cos(theta[1])*ufl.cos(theta[0])])
    d = ufl.dot(R0, Lm3)
    return d


# %% [markdown]
# In our 5-parameter Naghdi shell model the configuration of the shell is assigned by:
# - the 3-component vector field $\vec{u}$ representing the displacement with respect to the initial configuration
# $\vec{\phi}_0$
# - the 2-component vector field $\vec{\theta}$ representing the angle variation of the director $\vec{d}$ with respect
# to initial unit normal $\vec{n}$
#
# %% [markdown]
# Following [1], we use a $[P_2 + B_3]³$ element for $\vec{u}$ and a $[P_2]²$ element for $\vec{\theta}$ and collect
# them in the state vector $\vec{q} = [\vec{u}, \vec{\theta}]$:
#
# %%
# for the 3 translation DOFs, we use the P2 + B3 enriched element
P2 = FiniteElement("Lagrange", ufl.triangle, degree=2)
B3 = FiniteElement("Bubble", ufl.triangle, degree=3)
# Enriched
P2B3 = P2 + B3
# for 2 rotation DOFs, we use P2 element
# mixed element for u and theta
naghdi_shell_element = MixedElement([VectorElement(P2B3, dim=3), VectorElement(P2, dim=2)])
naghdi_shell_FS = FunctionSpace(mesh, naghdi_shell_element)

# %% [markdown]
# Then, we define `Function`, `TrialFunction` and `TestFunction` objects to express the variational forms and we split
# the mixed function into two subfunctions for displacement and rotation.

# %%
q_func = Function(naghdi_shell_FS)  # current configuration
q_trial = ufl.TrialFunction(naghdi_shell_FS)
q_test = ufl.TestFunction(naghdi_shell_FS)

u_func, theta_func = split(q_func)  # current displacement and rotation

# %% [markdown]
# We calculate the deformation gradient and the first, second fundamental forms:
#
# - Deformation gradient $\mathbf{F}$
#
# $$
# \mathbf{F} = \nabla \vec{\phi} \quad  (F_{ij} = \frac{\partial \phi_i}{\partial \xi_j}); \quad \vec{\phi} =
# \vec{\phi}_0 +
# \vec{u} \quad i = 1,2,3; j = 1,2
# $$
#
# - Metric tensor $\mathbf{a} \in \mathbb{S}^2_+$ and curvature tensor $\mathbf{b} \in \mathbb{S}^2$ (First and second
# fundamental form)
#
# $$
# \mathbf{a} = {\nabla \vec{\phi}} ^{T} \nabla \vec{\phi} \\
# \mathbf{b} = -\frac{1}{2}({\nabla \vec{\phi}} ^{T} \nabla \vec{d} + {\nabla \vec{d}} ^{T} \nabla \vec{\phi})
# $$
#
# In the initial configuration, $\vec{d} = \vec{n}$, $\vec{\phi} = \vec{\phi}_0$, the conresponding initial tensors are
# $\mathbf{a}_0$, $\mathbf{b}_0$

# %%
# current deformation gradient
F = grad(u_func) + grad(phi0_ufl)

# current director
d = director(R0_ufl, theta_func)

# initial metric and curvature tensor a0 and b0
a0_ufl = grad(phi0_ufl).T * grad(phi0_ufl)
b0_ufl = -0.5*(grad(phi0_ufl).T * grad(n0_ufl) + grad(n0_ufl).T * grad(phi0_ufl))

# %% [markdown]
# We define strain measures of the Naghdi shell model:
# - Membrane strain tensor $\boldsymbol{\varepsilon}(\vec{u})$
#
# $$
# \boldsymbol{\varepsilon} (\vec{u})= \frac{1}{2} \left ( \mathbf{a}(\vec{u}) - \mathbf{a}_0 \right)
# $$
#
# - Bending strain tensor $\boldsymbol{\kappa}(\vec{u}, \vec{\theta})$
#
# $$
# \boldsymbol{\kappa}(\vec{u}, \vec{\theta}) = \mathbf{b}(\vec{u}, \vec{\theta}) - \mathbf{b}_0
# $$
#
# - transverse shear strain vector $\vec{\gamma}(\vec{u}, \vec{\theta})$
#
# $$
# \begin{aligned}
# \vec{\gamma}(\vec{u}, \vec{\theta}) & = {\nabla \vec{\phi}(\vec{u})}^T \vec{d}(\vec{\theta}) - {\nabla\vec{\phi}_0}^T
# \vec{n} \\
# & = {\nabla \vec{\phi}(\vec{u})}^T \vec{d}(\vec{\theta}) \quad \text{if zero initial shears}
# \end{aligned}
# $$
#

# %%


# membrane strain
def epsilon(F):
    return 0.5 * (F.T * F - a0_ufl)


# bending strain
def kappa(F, d):
    return -0.5 * (F.T * grad(d) + grad(d).T * F) - b0_ufl


# transverse shear strain (zero initial shear strain)
def gamma(F, d):
    return F.T * d


# %% [markdown]
# In curvilinear coordinates, the stiffness modulus of linear isotropic material is defined as:
# - Membrane stiffness modulus $A^{\alpha\beta\sigma\tau}$, $D^{\alpha\beta\sigma\tau}$ (contravariant components)
#
# $$
# \frac{A^{\alpha\beta\sigma\tau}}t=12\frac{D^{\alpha\beta\sigma\tau}}{t^3}=\frac{2\lambda\mu}{\lambda+2\mu}
# a_0^{\alpha\beta}a_0^{\sigma\tau}+\mu(a_0^{\alpha\sigma}a_0^{\beta\tau}+a_0^{\alpha\tau}a_0^{\beta\sigma})
# $$
#
# - Shear stiffness modulus $S^{\alpha\beta}$ (contravariant components)
#
# $$
# \frac{S^{\alpha\beta}}t = \alpha_s \mu a_0^{\alpha\beta} , \quad \alpha_s = \frac{5}{6}: \text{shear factor}
# $$
#
# where $a_0^{\alpha\beta}$ is the contravariant components of the initial metric tensor $\mathbf{a}_0$

# %%
a0_contra_ufl = ufl.inv(a0_ufl)
j0_ufl = ufl.det(a0_ufl)

i, j, l, m = ufl.indices(4)
A_contra_ufl = ufl.as_tensor(
    (((2.0 * lmbda * mu) / (lmbda + 2.0 * mu)) * a0_contra_ufl[i, j] * a0_contra_ufl[l, m]
        + 1.0 * mu * (a0_contra_ufl[i, l] * a0_contra_ufl[j, m] + a0_contra_ufl[i, m] * a0_contra_ufl[j, l])),
    [i, j, l, m]
)

# %% [markdown]
# We define the resultant stress measures:
#
# - Membrane stress tensor $\mathbf{N}$
#
# $$
# \mathbf{N} = \mathbf{A} : \boldsymbol{\varepsilon}
# $$
#
# - Bending stress tensor $\mathbf{M}$
#
# $$
# \mathbf{M} = \mathbf{D} : \boldsymbol{\kappa}
# $$
#
# - Shear stress vector $\vec{T}$
#
# $$
# \vec{T} = \mathbf{S} \cdot \vec{\gamma}
# $$
#

# %%
N = ufl.as_tensor(t * A_contra_ufl[i, j, l, m] * epsilon(F)[l, m], [i, j])

M = ufl.as_tensor((t**3 / 12.0) * A_contra_ufl[i, j, l, m] * kappa(F, d)[l, m], [i, j])

T = ufl.as_tensor((t * mu * 5.0 / 6.0) * a0_contra_ufl[i, j] * gamma(F, d)[j], [i])

# %% [markdown]
# We define elastic strain energy density $\psi_{m}$, $\psi_{b}$, $\psi_{s}$ for membrane, bending and shear,
# respectively.
#
# $$
# \psi_m = \frac{1}{2} \mathbf{N} : \boldsymbol{\varepsilon}; \quad
# \psi_b = \frac{1}{2} \mathbf{M} : \boldsymbol{\kappa}; \quad
# \psi_s = \frac{1}{2} \vec{T} \cdot \vec{\gamma}
# $$
#
# They are per unit surface in the initial configuration:

# %%
psi_m = 0.5*inner(N, epsilon(F))

psi_b = 0.5*inner(M, kappa(F, d))

psi_s = 0.5*inner(T, gamma(F, d))

# %% [markdown]
# Shear and membrane locking is treated using the partial reduced selective integration proposed in Arnold and Brezzi
# [2].
#
# We introduce a parameter $\alpha \in \mathbb{R}$ that splits the membrane and shear energy in the energy functional
# into a weighted sum of two parts:
#
# $$
# \begin{aligned}\Pi_{N}(u,\theta)&=\Pi^b(u_h,\theta_h)+\alpha\Pi^m(u_h)+(1-\alpha)\Pi^m(u_h)\\&+\alpha\Pi^s(u_h,\theta_h)
# +(1-\alpha)\Pi^s(u_h,\theta_h)-W_{\mathrm{ext}},\end{aligned}
# $$
#
# We apply reduced integration to the parts weighted by the factor $(1-\alpha)$
#
# More details:
# - Optimal choice $\alpha = \frac{t^2}{h^2}$, $h$ is the diameter of the cell
# - Full integration : Gauss quadrature of degree 4 (6 integral points for triangle)
# - Reduced integration : Gauss quadrature of degree 2 (3 integral points for triangle).
#     - While [1] suggests a 1-point reduced integration, we observed that this leads to spurious modes in the present
# case.
#
#

# %%
# Full integration of order 4
dx_f = ufl.Measure('dx', domain=mesh, metadata={"quadrature_degree": 4})

# Reduced integration of order 2
dx_r = ufl.Measure('dx', domain=mesh, metadata={"quadrature_degree": 2})

# Calculate the factor alpha as a function of the mesh size h
h = ufl.CellDiameter(mesh)
alpha_FS = FunctionSpace(mesh, FiniteElement("DG", ufl.triangle, 0))
alpha_expr = Expression(t**2 / h**2, alpha_FS.element.interpolation_points())
alpha = Function(alpha_FS)
alpha.interpolate(alpha_expr)

# Full integration part of the total elastic energy
Pi_PSRI = psi_b * ufl.sqrt(j0_ufl) * dx_f
Pi_PSRI += alpha * psi_m * ufl.sqrt(j0_ufl) * dx_f
Pi_PSRI += alpha * psi_s * ufl.sqrt(j0_ufl) * dx_f

# Reduced integration part of the total elastic energy
Pi_PSRI += (1.0 - alpha) * psi_m * ufl.sqrt(j0_ufl) * dx_r
Pi_PSRI += (1.0 - alpha) * psi_s * ufl.sqrt(j0_ufl) * dx_r

# external work part (zero in this case)
W_ext = 0.0
Pi_PSRI -= W_ext

# %% [markdown]
# The residual and jacobian are the first and second order derivatives of the total potential energy, respectively

# %%
Residual = ufl.derivative(Pi_PSRI, q_func, q_test)
Jacobian = ufl.derivative(Residual, q_func, q_trial)

# %% [markdown]
# Next, we prescribe the dirichlet boundary conditions:
# - fully clamped boundary conditions on the top boundary ($\xi_2 = 0$):
#     - $u_{1,2,3} = \theta_{1,2} = 0$

# %%


# clamped boundary condition
def clamped_boundary(x):
    return np.isclose(x[1], 0.0)


fdim = tdim - 1
clamped_facets = locate_entities_boundary(mesh, fdim, clamped_boundary)

u_FS, _ = naghdi_shell_FS.sub(0).collapse()
theta_FS, _ = naghdi_shell_FS.sub(1).collapse()

# u1, u2, u3 = 0 on the clamped boundary
u_clamped = Function(u_FS)  # default value is 0
clamped_dofs_u = locate_dofs_topological((naghdi_shell_FS.sub(0), u_FS), fdim, clamped_facets)
bc_clamped_u = dirichletbc(u_clamped, clamped_dofs_u, naghdi_shell_FS.sub(0))

# theta1, theta2 = 0 on the clamped boundary
theta_clamped = Function(theta_FS)  # default value is 0
clamped_dofs_theta = locate_dofs_topological((naghdi_shell_FS.sub(1), theta_FS), fdim, clamped_facets)
bc_clamped_theta = dirichletbc(theta_clamped, clamped_dofs_theta, naghdi_shell_FS.sub(1))

# %% [markdown]
# - symmetry boundary conditions on the left and right side ($\xi_1 = \pm \pi/2$):
#     - $u_3 = \theta_2 = 0$

# %%


# symmetry boundary condition
def symm_boundary(x):
    return np.isclose(abs(x[0]), np.pi/2)


symm_facets = locate_entities_boundary(mesh, fdim, symm_boundary)

# u3 = 0 on the symmetry boundary
symm_dofs_u = locate_dofs_topological((naghdi_shell_FS.sub(0).sub(2), u_FS.sub(2)), fdim, symm_facets)
bc_symm_u = dirichletbc(u_clamped, symm_dofs_u, naghdi_shell_FS.sub(0).sub(2))

# theta2 = 0 on the symmetry boundary
symm_dofs_theta = locate_dofs_topological((naghdi_shell_FS.sub(1).sub(1), theta_FS.sub(1)), fdim, symm_facets)
bc_symm_theta = dirichletbc(theta_clamped, symm_dofs_theta, naghdi_shell_FS.sub(1).sub(1))

# all together
bcs = [bc_clamped_u, bc_clamped_theta, bc_symm_u, bc_symm_theta]

# %% [markdown]
# The loading is exerted by a point force along the $z$ direction applied at the midpoint of the bottom boundary.
# Since `PointSource` function is not available by far in new FEniCSx, we achieve the same functionality according
# to the reply in [4]

# %%


def compute_cell_contributions(V, points):
    # Determine what process owns a point and what cells it lies within
    mesh = V.mesh
    _, _, owning_points, cells = dolfinx.cpp.geometry.determine_point_ownership(
        mesh._cpp_object, points, 1e-6)
    owning_points = np.asarray(owning_points).reshape(-1, 3)

    # Pull owning points back to reference cell
    mesh_nodes = mesh.geometry.x
    cmap = mesh.geometry.cmaps[0]
    ref_x = np.zeros((len(cells), mesh.geometry.dim),
                     dtype=mesh.geometry.x.dtype)
    for i, (point, cell) in enumerate(zip(owning_points, cells)):
        geom_dofs = mesh.geometry.dofmap[cell]
        ref_x[i] = cmap.pull_back(point.reshape(-1, 3), mesh_nodes[geom_dofs])

    # Create expression evaluating a trial function (i.e. just the basis function)
    u = ufl.TrialFunction(V.sub(0).sub(2))
    num_dofs = V.sub(0).sub(2).dofmap.dof_layout.num_dofs * V.sub(0).sub(2).dofmap.bs
    if len(cells) > 0:
        # NOTE: Expression lives on only this communicator rank
        expr = dolfinx.fem.Expression(u, ref_x, comm=MPI.COMM_SELF)
        values = expr.eval(mesh, np.asarray(cells, dtype=np.int32))

        # Strip out basis function values per cell
        basis_values = values[:num_dofs:num_dofs*len(cells)]
    else:
        basis_values = np.zeros(
            (0, num_dofs), dtype=dolfinx.default_scalar_type)
    return cells, basis_values


# %%
# Point source
if mesh.comm.rank == 0:
    points = np.array([[0.0, L, 0.0]], dtype=mesh.geometry.x.dtype)
else:
    points = np.zeros((0, 3), dtype=mesh.geometry.x.dtype)

cells, basis_values = compute_cell_contributions(naghdi_shell_FS, points)
# cells: the cells that contain the points
# basis_values: the basis function values at the points

# %% [markdown]
# We define a custom `NonlinearProblem` which is able to compute the point force

# %%


class NonlinearProblemPointSource(NonlinearProblem):
    def __init__(self, F: ufl.form.Form, u: _Function, bcs: typing.List[DirichletBC] = [],
                 J: ufl.form.Form = None, cells=[], basis_values=[], PS: float = 0.0):
        super().__init__(F, u, bcs, J)
        self.PS = PS
        self.cells = cells
        self.basis_values = basis_values
        self.function_space = u.function_space

    def F(self, x: PETSc.Vec, b: PETSc.Vec) -> None:
        # Reset the residual vector
        with b.localForm() as b_local:
            b_local.set(0.0)
        assemble_vector(b, self._L)

        # Add point source
        if len(self.cells) > 0:
            for cell, basis_value in zip(self.cells, self.basis_values):
                dofs = self.function_space.sub(0).sub(2).dofmap.cell_dofs(cell)
                with b.localForm() as b_local:
                    b_local.setValuesLocal(dofs, basis_value * self.PS, addv=PETSc.InsertMode.ADD_VALUES)
        # Apply boundary condition
        apply_lifting(b, [self._a], bcs=[self.bcs], x0=[x], scale=-1.0)
        b.ghostUpdate(addv=PETSc.InsertMode.ADD, mode=PETSc.ScatterMode.REVERSE)
        set_bc(b, self.bcs, x, -1.0)

# %% [markdown]
# We use a standard Newton solver and modify the linear solver in each Newton iteration


# %%
problem = NonlinearProblemPointSource(Residual, q_func, bcs, Jacobian, cells, basis_values)

solver = NewtonSolver(mesh.comm, problem)

# Set Newton solver options
solver.rtol = 1e-6
solver.atol = 1e-6
solver.max_it = 30
solver.convergence_criterion = "incremental"
solver.report = True

# Modify the linear solver in each Newton iteration
ksp = solver.krylov_solver
opts = PETSc.Options()
option_prefix = ksp.getOptionsPrefix()
opts[f"{option_prefix}ksp_type"] = "cg"
# opts[f"{option_prefix}pc_type"] = "gamg"
opts[f"{option_prefix}pc_factor_mat_solver_type"] = "mumps"
ksp.setFromOptions()

# %% [markdown]
# Finally, we can solve the quasi-static problem, incrementally increasing the loading from 0 to $2000N$

# %%
PS_diff = 50.0
n_step = 40

# store the displacement at the point load
if mesh.comm.rank == 0:
    u3_list = np.zeros(n_step + 1)
    PS_list = np.arange(0, PS_diff * (n_step + 1), PS_diff)

q_func.x.array[:] = 0.0

bb_point = np.array([[0.0, L, 0.0]], dtype=np.float64)

for i in range(1, n_step + 1):
    problem.PS = PS_diff * i
    n, converged = solver.solve(q_func)
    assert (converged)
    q_func.x.scatter_forward()
    if mesh.comm.rank == 0:
        print(f"Load step {i:d}, Number of iterations: {n:d}, Load: {problem.PS:.2f}", flush=True)
    # calculate u3 at the point load
    u3_bb = None
    u3_func = q_func.sub(0).sub(2).collapse()
    if len(cells) > 0:
        u3_bb = u3_func.eval(bb_point, cells[0])[0]
    u3_bb = mesh.comm.gather(u3_bb, root=0)
    if mesh.comm.rank == 0:
        for u3 in u3_bb:
            if u3 is not None:
                u3_list[i] = u3
                break

# %% [markdown]
# We write the outputs of $\vec{u}$, $\vec{\theta}$, and $\vec{\phi}$ in the second order Lagrange space.

# %%
# interpolate phi_ufl into CG2 Space
u_P2B3 = q_func.sub(0).collapse()
theta_P2 = q_func.sub(1).collapse()

# Interpolate phi in the [P2]³ Space
phi_FS = FunctionSpace(mesh, VectorElement("Lagrange", ufl.triangle, degree=2, dim=3))
phi_expr = Expression(phi0_ufl + u_P2B3, phi_FS.element.interpolation_points())
phi_func = Function(phi_FS)
phi_func.interpolate(phi_expr)

# Interpolate u in the [P2]³ Space
u_P2 = Function(phi_FS)
u_P2.interpolate(u_P2B3)

results_folder = Path("results/nonlinear_Naghdi/semi_cylinder")
results_folder.mkdir(exist_ok=True, parents=True)

with dolfinx.io.VTXWriter(mesh.comm, results_folder/"u_naghdi.bp", [u_P2]) as vtx:
    vtx.write(0)

with dolfinx.io.VTXWriter(mesh.comm, results_folder/"theta_naghdi.bp", [theta_P2]) as vtx:
    vtx.write(0)

with dolfinx.io.VTXWriter(mesh.comm, results_folder/"phi_naghdi.bp", [phi_func]) as vtx:
    vtx.write(0)

# %% [markdown]
# The results for the transverse displacement at the point of application of the force are validated against a standard
# reference from the literature, obtained using Abaqus S4R element and a structured mesh of 40×40 elements, see [1]:

# %%
if mesh.comm.rank == 0:
    fig = plt.figure(figsize=(8, 6))
    reference_u3 = 1.e-2*np.array([0., 5.421, 16.1, 22.195, 27.657, 32.7, 37.582, 42.633,
                                   48.537, 56.355, 66.410, 79.810, 94.669, 113.704, 124.751, 132.653,
                                   138.920, 144.185, 148.770, 152.863, 156.584, 160.015, 163.211,
                                   166.200, 168.973, 171.505])
    reference_P = 2000.*np.array([0., .05, .1, .125, .15, .175, .2, .225, .25, .275, .3,
                                  .325, .35, .4, .45, .5, .55, .6, .65, .7, .75, .8, .85, .9, .95, 1.])
    plt.plot(-u3_list, PS_list, label='FEniCSx-shell 20 x 20')
    plt.plot(reference_u3, reference_P, "or", label='Sze (Abaqus S4R)')
    plt.xlabel("Displacement (mm)")
    plt.ylabel("Load (N)")
    plt.legend()
    plt.grid()
    plt.savefig(results_folder/"comparisons.png")
# %% [markdown]
# References:
#
# [1] K. Sze, X. Liu, and S. Lo. Popular benchmark problems for geometric nonlinear analysis of shells. Finite Elements
# in Analysis and Design, 40(11):1551 – 1569, 2004.
#
# [2] D. Arnold and F.Brezzi, Mathematics of Computation, 66(217): 1-14, 1997.
# https://www.ima.umn.edu/~arnold//papers/shellelt.pdf
#
# [3] P. Betsch, A. Menzel, and E. Stein. On the parametrization of finite rotations in computational mechanics:
# A classification of concepts with application to smooth shells. Computer Methods in Applied Mechanics and Engineering,
# 155(3):273 – 305, 1998.
#
# [4] https://fenicsproject.discourse.group/t/point-sources-redux/13496/4
