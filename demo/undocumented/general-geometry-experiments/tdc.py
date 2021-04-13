import numpy as np

from mpi4py import MPI

import dolfinx
import ufl

domain = ufl.Mesh(ufl.VectorElement(
        "Lagrange", ufl.Cell("triangle", geometric_dimension=3), 1))
x = np.array([[0.0, 0.0, 0.0], [1.0, 0.0, 0.0], [1.0, 1.0, 0.0]])
cells = np.array([[0, 1, 2]], dtype=np.int64)
mesh = dolfinx.mesh.create_mesh(MPI.COMM_WORLD, cells, x, domain)

# Roughly follows notation from Michael Neunteufel's PhD thesis
tau = ufl.Jacobian(mesh)
tau_1 = tau[:, 0]
tau_2 = tau[:, 1]

t_12 = ufl.cross(tau_1, tau_2)
nu = t_12/ufl.sqrt(ufl.inner(t_12, t_12)) # Theorem 7.2.1

P = ufl.Identity(3) - ufl.outer(nu, nu) # Eq. 7.2.5

W_el = ufl.MixedElement([ufl.VectorElement("Lagrange", ufl.Cell("triangle", geometric_dimension=3), 1, dim=3),
                         ufl.VectorElement("Lagrange", ufl.Cell("triangle", geometric_dimension=3), 1, dim=3)])
W = dolfinx.FunctionSpace(mesh, W_el)
w = dolfinx.Function(W)

u, beta = w.split()

# Membrane energy
F_tau = ufl.grad(u) + P
C_tau = F_tau.T*F_tau
E_tau = 0.5*(C_tau - P)

# NOTE: Missing material properties
Pi_M = ufl.inner(E_tau, E_tau)

# Shear energy
E_S = ufl.grad(u).T*nu - beta
# NOTE: Missing material properties
Pi_S = ufl.inner(E_S, E_S)

# NOTE: Need to do patch averaging of unit normals to take ufl.grad(nu)
# Bending energy
# NOTE: Missing material properties
E_B = ufl.sym(P*ufl.grad(beta)) - ufl.sym(ufl.grad(u).T*ufl.grad(nu))
Pi_B = ufl.inner(E_B, E_B)

dolfinx.fem.assemble_scalar(Pi_M*ufl.dx + Pi_B*ufl.dx + Pi_S*ufl.dx)
