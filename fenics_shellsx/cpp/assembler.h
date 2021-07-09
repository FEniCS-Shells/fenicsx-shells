#include <iostream>

#include <pybind11/pybind11.h>

#include <Eigen/Dense>

#include <dolfinx/common/IndexMap.h>
#include <dolfinx/fem/DofMap.h>
#include <dolfinx/fem/FiniteElement.h>
#include <dolfinx/fem/Form.h>
#include <dolfinx/fem/petsc.h>
#include <dolfinx/fem/utils.h>
#include <dolfinx/fem/Constant.h>
#include <dolfinx/fem/FunctionSpace.h>
#include <dolfinx/la/PETScMatrix.h>
#include <dolfinx/la/PETScVector.h>
#include <dolfinx/la/SparsityPattern.h>
#include <dolfinx/mesh/Mesh.h>

// What is the proper way to get the location of these pybind11 casting headers?
#include "/usr/local/dolfinx-real/lib/python3.8/dist-packages/dolfinx/wrappers/caster_petsc.h"
#include <petscmat.h>

#include <xtensor/xtensor.hpp>

using namespace dolfinx;

// TODO:
// Boundary condition application (can it be done with the standard routines?)
// Convert to active_cells pattern as in standard assemblers.
// Work out how to get include directory of caster_petsc.h
// Split down into inlined functions for improved readability.

template <typename T>
std::pair<Mat, Vec> assemble(const fem::Form<T>& a, const fem::Form<T>& L)
{
  assert(a.rank() == 2);
  assert(L.rank() == 1);

  // Get data from form
  const auto mesh = a.mesh();
  assert(mesh);
  assert(mesh == L.mesh());

  const std::shared_ptr<const fem::DofMap> dofmap0
      = a.function_spaces().at(0)->dofmap();
  const std::shared_ptr<const fem::DofMap> dofmap1
      = a.function_spaces().at(1)->dofmap();
  // Assumption we are working with bilinear form leading to square operator.
  assert(dofmap0);
  assert(dofmap1);
  assert(dofmap0 == dofmap1);

  const std::shared_ptr<const fem::FiniteElement> element
      = a.function_spaces().at(0)->element();
  assert(element->num_sub_elements() == 2);

  const auto primal_dofmap = std::make_shared<const dolfinx::fem::DofMap>(
      dofmap0->extract_sub_dofmap({0}));
  const auto primal_element = element->extract_sub_element({0});

  // Make data structures for global assembly
  // Create sparsity pattern
  // std::array primal_dofmaps{primal_dofmap.get(), primal_dofmap.get()};
  const std::array primal_dofmaps{a.function_spaces().at(0)->dofmap().get(),
                                  a.function_spaces().at(1)->dofmap().get()};
  la::SparsityPattern pattern
      = fem::create_sparsity_pattern(a);
  pattern.assemble();

  auto A = la::PETScMatrix(mesh->mpi_comm(), pattern);
  auto b = la::PETScVector(*(primal_dofmap->index_map), 1);
  // Create mat_set_values function pointer
  std::function<int(std::int32_t, const std::int32_t*, std::int32_t,
                    const std::int32_t*, const T*)>
      mat_set_values = la::PETScMatrix::set_fn(A.mat(), InsertMode::ADD_VALUES);
  MatZeroEntries(A.mat());

  // Extract raw view into PETSc Vec memory.
  Vec b_local;
  VecGhostGetLocalForm(b.vec(), &b_local);
  PetscInt n = 0;
  VecGetSize(b_local, &n);
  PetscScalar* array = nullptr;
  VecGetArray(b_local, &array);
  Eigen::Map<Eigen::Matrix<PetscScalar, Eigen::Dynamic, 1>> _b(array, n);

  // Calculate offsets into larger local tensor for slicing.
  auto dual_element = element->extract_sub_element({1});
  assert(dual_element->num_sub_elements() == 2);
  auto offsets = Eigen::Array<int, 3, 1>();
  auto sizes = Eigen::Array<int, 3, 1>();

  offsets[0] = 0;
  offsets[1] = offsets[0] + primal_element->space_dimension();
  offsets[2]
      = offsets[1] + dual_element->extract_sub_element({0})->space_dimension();

  sizes[0] = offsets[1] - offsets[0];
  sizes[1] = offsets[2] - offsets[1];
  sizes[2] = element->space_dimension() - offsets[2];

  // Local tensors
  Eigen::Matrix<T, Eigen::Dynamic, Eigen::Dynamic, Eigen::RowMajor> Ae(
      element->space_dimension(), element->space_dimension());
  Eigen::Matrix<T, Eigen::Dynamic, 1> be(element->space_dimension());

  Eigen::Matrix<T, Eigen::Dynamic, Eigen::Dynamic, Eigen::RowMajor> A_macro(
      2 * element->space_dimension(), 2 * element->space_dimension());
  Eigen::Matrix<T, Eigen::Dynamic, 1> b_macro(2 * element->space_dimension());

  // Block splitting (views into Ae and be).
  const auto A_00 = Ae.block(offsets[0], offsets[0], sizes[0], sizes[0]);
  const auto A_02 = Ae.block(offsets[0], offsets[2], sizes[0], sizes[2]);
  const auto A_11 = Ae.block(offsets[1], offsets[1], sizes[1], sizes[1]);
  const auto A_12 = Ae.block(offsets[1], offsets[2], sizes[1], sizes[2]);
  const auto A_20 = Ae.block(offsets[2], offsets[0], sizes[2], sizes[0]);
  const auto A_21 = Ae.block(offsets[2], offsets[1], sizes[2], sizes[1]);

  const auto b_0 = be.segment(offsets[0], sizes[0]);
  const auto b_1 = be.segment(offsets[1], sizes[1]);
  const auto b_2 = be.segment(offsets[2], sizes[2]);

  // Iterate over active cells
  const int tdim = mesh->topology().dim();
  const auto map = mesh->topology().index_map(tdim);
  assert(map);
  const int num_cells = map->size_local();

  using type = fem::IntegralType;

  assert(a.num_integrals(type::cell) == 1);
  assert(a.num_integrals(type::interior_facet) == 1);
  assert(a.num_integrals(type::exterior_facet) == 1);
  const auto a_kernel_domain_integral
      = a.kernel(type::cell, -1);
  const auto a_kernel_interior_facet
      = a.kernel(type::interior_facet, -1);
  const auto a_kernel_exterior_facet
      = a.kernel(type::exterior_facet, -1);
  assert(L.num_integrals(type::cell) == 1);
  assert(L.num_integrals(type::interior_facet) == 1);
  assert(L.num_integrals(type::exterior_facet) == 1);
  const auto L_kernel_domain_integral
      = L.kernel(type::cell, -1);
  const auto L_kernel_interior_facet
      = L.kernel(type::interior_facet, -1);
  const auto L_kernel_exterior_facet
      = L.kernel(type::exterior_facet, -1);

  // Prepare cell geometry
  const int gdim = mesh->geometry().dim();
  const graph::AdjacencyList<std::int32_t>& x_dofmap
      = mesh->geometry().dofmap();

  // FIXME: Add proper interface for num coordinate dofs
  const std::size_t num_dofs_g = x_dofmap.num_links(0);
  const xt::xtensor<double, 2>& x_g
      = mesh->geometry().x();
  std::vector<double> coordinate_dofs(num_dofs_g * 3);
  xt::xtensor<double, 3> coordinate_dofs_macro({2, num_dofs_g, 3});

  // Prepare constants
  const auto& a_constants
      = fem::pack_constants(a);
  const auto& L_constants
      = fem::pack_constants(L);

  // Prepare coefficients
  const auto&
      a_coeffs = pack_coefficients(a);
  const auto&
      L_coeffs = pack_coefficients(L);

  const std::vector<int> a_offsets = a.coefficient_offsets();
  std::vector<T> a_coeff_array_macro(2 * a_offsets.back());
  const std::vector<int> L_offsets = L.coefficient_offsets();
  std::vector<T> L_coeff_array_macro(2 * L_offsets.back());

  // Needed for all integrals
  mesh->topology_mutable().create_entity_permutations();
  const std::vector<unsigned int>& cell_info
      = mesh->topology().get_cell_permutation_info();

  // Needed for facet integrals
  const std::vector<std::uint8_t>& perms
      = mesh->topology().get_facet_permutations();

  // For static condensation step
  Eigen::Array<T, Eigen::Dynamic, Eigen::Dynamic, Eigen::RowMajor> Ae_projected(
      primal_element->space_dimension(), primal_element->space_dimension());
  Eigen::Array<T, Eigen::Dynamic, 1> be_projected(
      primal_element->space_dimension());

  mesh->topology_mutable().create_connectivity(tdim - 1, tdim);
  mesh->topology_mutable().create_connectivity(tdim, tdim - 1);

  const auto f_to_c = mesh->topology().connectivity(tdim - 1, tdim);
  assert(f_to_c);
  const auto c_to_f = mesh->topology().connectivity(tdim, tdim - 1);
  assert(c_to_f);

  for (int c = 0; c < num_cells; ++c)
  {
    // Get cell vertex coordinates
    auto x_dofs = x_dofmap.links(c);
    for (std::size_t i = 0; i < x_dofs.size(); ++i)
    {
      std::copy_n(xt::row(x_g, x_dofs[i]).begin(), 3,
                  std::next(coordinate_dofs.begin(), i * 3));
    };

    Ae.setZero();
    be.setZero();

    const auto a_coeff_array = a_coeffs.row(c);
    const auto L_coeff_array = L_coeffs.row(c);
    a_kernel_domain_integral(Ae.data(), a_coeff_array.data(),
                             a_constants.data(), coordinate_dofs.data(),
                             nullptr, nullptr);
    L_kernel_domain_integral(be.data(), L_coeff_array.data(),
                             L_constants.data(), coordinate_dofs.data(),
                             nullptr, nullptr);

    // Loop over attached facets
    const auto c_f = c_to_f->links(c);
    assert(c_to_f->num_links(c) == 3);

    for (int local_facet = 0; local_facet < 3; ++local_facet)
    {
      // Get attached cell indices
      const std::int32_t f = c_f[local_facet];
      const auto f_c = f_to_c->links(f);
      assert(f_c.size() < 3);

      if (f_c.size() == 1)
      {
        // Exterior facet
        // Get local index of facet with respect to cell.
        const std::uint8_t perm = perms[c * c_f.size() + local_facet];
        a_kernel_exterior_facet(Ae.data(), a_coeff_array.data(),
                                a_constants.data(), coordinate_dofs.data(),
                                &local_facet, &perm);
        L_kernel_exterior_facet(be.data(), L_coeff_array.data(),
                                L_constants.data(), coordinate_dofs.data(),
                                &local_facet, &perm);
      }
      else
      {
        // Interior facet
        // Create attached cells
        // Find local facet numbering
        std::array<int, 2> local_facets;
        for (int k = 0; k < 2; ++k)
        {
          const auto c_f = c_to_f->links(f_c[k]);
          const auto* end = c_f.data() + c_f.size();
          const auto* it = std::find(c_f.data(), end, f);
          assert(it != end);
          local_facets[k] = std::distance(c_f.data(), it);
        }

        // Orientation
        const std::array perm{perms[f_c[0] * c_f.size() + local_facets[0]],
                              perms[f_c[1] * c_f.size() + local_facets[1]]};

        // Get cell geometry
        auto x_dofs0 = x_dofmap.links(f_c[0]);
        for (std::size_t i = 0; i < x_dofs0.size(); ++i)
        {
          std::copy_n(xt::view(x_g, x_dofs0[i]).begin(), 3,
                      xt::view(coordinate_dofs_macro, 0, i, xt::all()).begin());
        }
        auto x_dofs1 = x_dofmap.links(f_c[1]);
        for (std::size_t i = 0; i < x_dofs1.size(); ++i)
        {
          std::copy_n(xt::view(x_g, x_dofs1[i]).begin(), 3,
                      xt::view(coordinate_dofs_macro, 1, i, xt::all()).begin());
        }       // Get cell geometry

        // Layout for the restricted coefficients is flattened
        // w[coefficient][restriction][dof]
        const auto a_coeff_cell0 = a_coeffs.row(f_c[0]);
        const auto a_coeff_cell1 = a_coeffs.row(f_c[1]);
        const auto L_coeff_cell0 = L_coeffs.row(f_c[0]);
        const auto L_coeff_cell1 = L_coeffs.row(f_c[1]);

        // Loop over coefficients for a
        for (std::size_t i = 0; i < a_offsets.size() - 1; ++i)
        {
          // Loop over entries for coefficient i
          const int num_entries = a_offsets[i + 1] - a_offsets[i];
          std::copy_n(a_coeff_cell0.data() + a_offsets[i], num_entries,
                      std::next(a_coeff_array_macro.begin(), 2 * a_offsets[i]));
          std::copy_n(a_coeff_cell1.data() + a_offsets[i], num_entries,
                      std::next(a_coeff_array_macro.begin(),
                                a_offsets[i + 1] + a_offsets[i]));
        }

        // Loop over coefficients for L
        for (std::size_t i = 0; i < L_offsets.size() - 1; ++i)
        {
          // Loop over entries for coefficient i
          const int num_entries = L_offsets[i + 1] - L_offsets[i];
          std::copy_n(L_coeff_cell0.data() + L_offsets[i], num_entries,
                      std::next(L_coeff_array_macro.begin(), 2 * L_offsets[i]));
          std::copy_n(L_coeff_cell1.data() + L_offsets[i], num_entries,
                      std::next(L_coeff_array_macro.begin(),
                                L_offsets[i + 1] + L_offsets[i]));
        }


        A_macro.setZero();
        b_macro.setZero();

        a_kernel_interior_facet(
            A_macro.data(), a_coeff_array_macro.data(), a_constants.data(),
            coordinate_dofs_macro.data(), local_facets.data(), perm.data());
        L_kernel_interior_facet(
            b_macro.data(), L_coeff_array_macro.data(), L_constants.data(),
            coordinate_dofs_macro.data(), local_facets.data(), perm.data());

        // Assemble appropriate part of A_macro/b_macro into Ae/be.
        int local_cell = (f_c[0] == c ? 0 : 1);
        int offset = local_cell * element->space_dimension();
        Ae += A_macro.block(offset, offset, Ae.rows(), Ae.cols());
        be += b_macro.block(offset, 0, be.rows(), 1);
      }
    }
    // TODO: Apply boundary conditions.

    // Perform static condensation.
    // TODO: Two calls to same .inverse().
    // Verified that Ae_projected is the same as old fenics-shells code.
    Ae_projected = A_00 + A_02 * A_12.inverse() * A_11 * A_21.inverse() * A_20;
    be_projected = b_0 + A_02 * A_12.inverse() * A_11 * A_21.inverse() * b_2
                   - A_02 * A_12.inverse() * b_1;

    // Assembly.
    const auto dofs = primal_dofmap->list().links(c);
    mat_set_values(dofs.size(), dofs.data(), dofs.size(), dofs.data(),
                   Ae_projected.data());

    for (Eigen::Index i = 0; i < dofs.size(); ++i)
      _b[dofs[i]] += be_projected[i];
  }

  VecRestoreArray(b_local, &array);
  VecGhostRestoreLocalForm(b.vec(), &b_local);

  // The DOLFINX la::PETscVector and la::PETScMatrix objects dereference the
  // underlying PETSc Vec and Mat objects when they go out of scope. So, we
  // need to increment the PETSc reference counter before passing them out
  // to conversion by pybind11 to petsc4py wrapped objects.
  PetscObjectReference((PetscObject)A.mat());
  PetscObjectReference((PetscObject)b.vec());
  return std::make_pair<Mat, Vec>(A.mat(), b.vec());
}
