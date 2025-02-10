
from mpi4py import MPI
import dolfinx
import numpy as np
import basix.ufl
import ufl
from petsc4py import PETSc
import math
import numpy.typing as npt
#from projector import Projector

import numba

from dolfinx.fem.petsc import assemble_matrix
from dolfinx.utils import numba_utils as petsc_numba
from dolfinx.utils import cffi_utils as petsc_cffi
from dolfinx.utils import ctypes_utils as petsc_ctypes
from dolfinx import default_scalar_type
try:
    MatSetValuesLocal = petsc_numba.MatSetValuesLocal
    MatSetValuesLocal_ctypes = petsc_ctypes.MatSetValuesLocal
    MatSetValuesLocal_abi = petsc_cffi.MatSetValuesLocal
except AttributeError:
    MatSetValuesLocal_abi = None
    
petsc_options={"ksp_type": "preonly", "pc_type": "lu"}

def unroll_dofmap(dofs: npt.NDArray[np.int32], bs: int) -> npt.NDArray[np.int32]:
    """
    Given a two-dimensional dofmap of size `(num_cells, num_dofs_per_cell)`
    Expand the dofmap by its block size such that the resulting array
    is of size `(num_cells, bs*num_dofs_per_cell)`
    """
    num_cells, num_dofs_per_cell = dofs.shape
    unrolled_dofmap = np.repeat(dofs, bs).reshape(num_cells, num_dofs_per_cell * bs) * bs
    unrolled_dofmap += np.tile(np.arange(bs), num_dofs_per_cell)
    return unrolled_dofmap

@numba.njit
def sink(*args):
    pass

@numba.njit
def set_vals_numba(A, m, rows, n, cols, data, mode):
    MatSetValuesLocal(A, 3, rows.ctypes, 3, cols.ctypes, data.ctypes, mode)


# Bernoulli function 
@numba.njit(fastmath=True)
def bernoulli(r) -> float:
    if np.absolute(r) < 1e-10:
        return 1.0

    if r < 0.0:
        return r / np.expm1(r)

    return r * np.exp(-r) / (1 - np.exp(-r))
    
    
mesh = dolfinx.mesh.create_unit_square(MPI.COMM_WORLD, 1, 1)
V = dolfinx.fem.functionspace(mesh, ("CG", 1))

u = ufl.TrialFunction(V)
v = ufl.TestFunction(V)

cell_map = mesh.topology.index_map(2)
num_cells = cell_map.size_local 

v_map = V.dofmap.list
unrolled_map = unroll_dofmap(v_map, V.dofmap.index_map_bs)

x_dofs = mesh.geometry.dofmap
x = mesh.geometry.x

#Quadrature rule for edge  
q_facet, q_facetw = basix.quadrature.make_quadrature(basix.CellType.interval, 1, basix.quadrature.QuadratureType.default)

#Quadrature rule for triangle 
q_cell, q_cellw = basix.quadrature.make_quadrature(basix.CellType.triangle, 2, basix.quadrature.QuadratureType.default) 

# Quadrature rule for the jacobian.
q_jaco, q_jacow = basix.quadrature.make_quadrature(basix.CellType.triangle, 1, basix.quadrature.QuadratureType.default)

# Build the bilinear matrix 
mass_form = dolfinx.fem.form(ufl.inner(u, v)*ufl.dx)
M = assemble_matrix(mass_form)
M.assemble()
A1 = M.copy()
A1.zeroEntries()

# 
B = dolfinx.fem.functionspace(mesh, ("DG", 0, (2, )))
b = dolfinx.fem.Function(B)
b.interpolate(lambda x: (x[0], x[1]))


diff = dolfinx.fem.Function(V)
diff.interpolate(lambda x: 1+x[0]*x[1] )

# Define expressions, note different quadrature rules 
detJ_expr      = dolfinx.fem.Expression(ufl.det(ufl.Jacobian(mesh)), q_jaco, comm=mesh.comm)
trial_expr     = dolfinx.fem.Expression(u, q_cell, comm=mesh.comm)
gradtrial_expr = dolfinx.fem.Expression(ufl.grad(u), q_cell, comm=mesh.comm)
alpha_expr =  dolfinx.fem.Expression( diff, q_jaco, comm=mesh.comm)


A_local = np.empty((3, 3), dtype=PETSc.ScalarType)

# Eval expression on all cells.
detJ = detJ_expr.eval(mesh, np.arange(num_cells, dtype=np.int32) )
phi  = trial_expr.eval(mesh, np.arange(num_cells, dtype=np.int32) ).reshape(num_cells,3,3) 
gphi = gradtrial_expr.eval(mesh, np.arange(num_cells, dtype=np.int32)).reshape(num_cells,6,3)

cellwise_alpha = alpha_expr.eval( mesh, np.arange(num_cells, dtype=np.int32))


# Follows implementation of https://github.com/pdelab/pyeafe
 

for cell in range(num_cells):    
    # Vertex dofs : global indices  
    V_dofs = unrolled_map[cell]
    
    # DG0 Vector Function value 
    b_values = b.x.array[2*cell:2*cell+2]
    # Reset local matrix 
    A_local[:] = 0.0
    
    # Cellwise alpha
    alpha = cellwise_alpha[cell]
    
    # Local stiffness matrix 
    for j in range(q_cell.shape[0]): 
        M = gphi[cell,2*j:2*(j+1)]
        for row in range(3):
           for col in range(3):
              A_local[row,col] += q_cellw[j]*(M[0,row]*M[0,col] + M[1,row]*M[1,col]) 
   
    # Set diagonal to zero  
    for row in range(3): 
        A_local[row,row]  = 0

    # Local mass lumping   
    for j in range(q_cell.shape[0]): 
        N = phi[cell,j,:]
        for row in range(3):
            for col in range(3):
                A_local[row,row] += q_cellw[j]*N[row]*N[col]     
          
    A_local*=abs(detJ[cell])
     
    for i in range(3):
        # Facet 0 has vertex 1 and 2, etc.
        k = (i+1)%3
        l = (i+2)%3
        # Global vertex indicies 
        gl, gk = V_dofs[l],V_dofs[k] 
        # Vertex coordinates  
        xi, xj = x[gl],x[gk] 
        # 
        value = np.dot(b_values, (xi-xj)[0:2])
        # 
        A_local[k,l] *= alpha*bernoulli(  value ) 
        A_local[k,k] -= A_local[k,l]
        A_local[l,k] *= alpha*bernoulli( -value )
        A_local[l,l] -= A_local[l,k]

   
    set_vals_numba(A1.handle, 3, V_dofs, 3, V_dofs, A_local, PETSc.InsertMode.ADD_VALUES)



A1.assemble()
A1.convert("dense")
print( A1.getDenseArray() ) 






