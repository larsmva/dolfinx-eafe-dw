
from mpi4py import MPI
import dolfinx
import numpy as np
import basix.ufl
import ufl
from petsc4py import PETSc
import math
import numpy.typing as npt
from projector import Projector

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

@numba.njit
def sink(*args):
    pass

@numba.njit
def set_vals_numba(A, m, rows, n, cols, data, mode):
    MatSetValuesLocal(A, 3, rows.ctypes, 3, cols.ctypes, data.ctypes, mode)


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


@numba.njit(fastmath=True)
def bernoulli(r) -> float:
    if np.absolute(r) < 1e-10:
        return 1.0

    if r < 0.0:
        return r / np.expm1(r)

    return r * np.exp(-r) / (1 - np.exp(-r))
    
    
@numba.njit(fastmath=True)
def assemble_eafe( A_global, x, dofmap, phi, grad_phi, detJ, cellwise_alpha, cellwise_psi, mode ): 

   A_local = np.empty((3, 3), dtype=PETSc.ScalarType)
   num_cells = dofmap.shape[0]
   q_cell  = np.array([[1./6, 1/6], [1/6, 2/3], [2/3, 1/6]], dtype=np.double)
   q_cellw = np.full(3, 1.0 / 6.0, dtype=np.double)



   for cell in range(num_cells):    
        # Vertex dofs : global indices  
        V_dofs = dofmap[cell]
        
        cell_area = np.abs(detJ[cell])
        # DG 0 Function values corresponding to cell 
        psi = cellwise_psi[cell]
        
        alpha = cellwise_alpha[cell]
        # Reset local matrix 
        A_local[:] = 0.0
    
    
        # Local stiffness matrix 
        for j in range(q_cell.shape[0]): 
            M = grad_phi[cell,2*j :2*(j+1),:]
            for row in range(3):
                for col in range(3):
                    A_local[row,col] += q_cellw[j]*(M[0,row]*M[0,col] + M[1,row]*M[1,col]) 

      
        np.fill_diagonal(A_local,0.0)

        # Local mass lumping   
        for j in range(q_cell.shape[0]): 
            N = phi[cell,j,:]
            for row in range(3):
                for col in range(3):
                    A_local[row,row] += q_cellw[j]*N[row]*N[col]    
        
        A_local*=cell_area          
        for i in range(3):
            k = (i+1)%3
            l = (i+2)%3
        
            gl, gk = V_dofs[l],V_dofs[k] 

            xi, xj = x[gl],x[gk] 

            value = np.dot(psi, (xi-xj)[0:2])

            A_local[k,l] *= alpha*bernoulli( value ) 
            A_local[k,k] -= A_local[k,l]
            A_local[l,k] *= alpha*bernoulli( -value )
            A_local[l,l] -= A_local[l,k]


        pos = V_dofs
        set_vals_numba(A_global, 3, pos, 3, pos, A_local, mode)
   sink(A_local,dofmap)


"""
    Direct implementation of aefe. 
    
    
    int_{T} J_T(u)*grad(v) dx ~ Sum_E 
    


"""

   




@numba.njit(fastmath=True)
def assemble_dw_lhs( A_global, x, dofmap, phi, grad_phi, detJ, cellwise_alpha, cellwise_psi, mode ): 

   A_local = np.empty((3, 3), dtype=PETSc.ScalarType)
   num_cells = dofmap.shape[0]
   q_cell  = np.array([[1./6, 1/6], [1/6, 2/3], [2/3, 1/6]], dtype=np.double)
   q_cellw = np.full(3, 1.0 / 6.0, dtype=np.double)

   for cell in range(num_cells):    
        # Vertex dofs : global indices  
        V_dofs = dofmap[cell]
        
        cell_area = np.abs(detJ[cell])
        # DG 0 Function values corresponding to cell 
        psi = cellwise_psi[cell]
        
        alpha = cellwise_alpha[cell]
        # Reset local matrix 
        A_local[:] = 0.0
    
    
        # Local stiffness matrix 
        for j in range(q_cell.shape[0]): 
            M = grad_phi[cell,2*j :2*(j+1),:]
            for row in range(3):
                for col in range(3):
                    A_local[row,col] += q_cellw[j]*(M[0,row]*M[0,col] + M[1,row]*M[1,col]) 

      
        np.fill_diagonal(A_local,0.0)

        # Local mass lumping   
        for j in range(q_cell.shape[0]): 
            N = phi[cell,j,:]
            for row in range(3):
                for col in range(3):
                    A_local[row,row] += 2*q_cellw[j]*N[row]*N[col]    
        
        A_local*=cell_area          
        for i in range(3):
            k = (i+1)%3
            l = (i+2)%3
        
            gl, gk = V_dofs[l],V_dofs[k] 

            xi, xj = x[gl],x[gk] 

            value = np.dot(psi, (xi-xj)[0:2])

            A_local[k,l] *= alpha*bernoulli( value ) 
            A_local[k,k] -= A_local[k,l]
            A_local[l,k] *= alpha*bernoulli( -value )
            A_local[l,l] -= A_local[l,k]

       
        set_vals_numba(A_global, 3, V_dofs, 3, V_dofs , A_local, mode)
   sink(A_local,dofmap)


@numba.njit(fastmath=True)
def assemble_edge_direction(A_global, x, dofmap, detJ, mode):
   A_local = np.empty((3, 3), dtype=PETSc.ScalarType)
   num_cells = dofmap.shape[0]
   q_cell  = np.array([[1./6, 1/6], [1/6, 2/3], [2/3, 1/6]], dtype=np.double)
   q_cellw = np.full(3, 1.0 / 6.0, dtype=np.double)

   for cell in range(num_cells):    
        # Vertex dofs : global indices  
        V_dofs = dofmap[cell]
        
        cell_area = np.abs(detJ[cell])
        # DG 0 Function values corresponding to cell 
        
        A_local[:] = 0.0
    
        for i in range(3):
            k = (i+1)%3
            l = (i+2)%3
        
            gl, gk = V_dofs[l],V_dofs[k] 

            xi, xj = x[gl],x[gk] 

            value = np.dot(psi, (xi-xj)[0:2])

            A_local[k,l] += value  
            A_local[l,k] += -value 
   sink(A_local,dofmap)
   
   






