
from mpi4py import MPI
import dolfinx
import numpy as np
import basix.ufl
import ufl
from petsc4py import PETSc
import math
import numpy.typing as npt
from projector import Projector

#
from assemble_eafe import * 

import numba

from dolfinx.fem.petsc import assemble_matrix, assemble_vector
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


def neatprint(array ): 
       for row in array:
             print(*['{:2.1f}\t'.format(each) for each in row])
   
   
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
    
    
mesh = dolfinx.mesh.create_unit_square(MPI.COMM_WORLD, 3, 3)
V = dolfinx.fem.functionspace(mesh, ("CG", 1))
u = ufl.TrialFunction(V)
v = ufl.TestFunction(V)

cell_map = mesh.topology.index_map(2)
num_cells = cell_map.size_local + cell_map.num_ghosts

v_map = V.dofmap.list
unrolled_map = unroll_dofmap(v_map, V.dofmap.index_map_bs)

x_dofs = mesh.geometry.dofmap
x = mesh.geometry.x


def smooth_cutoff_func(d, eps: float = 1e-4) :
    return ufl.conditional( d >eps, d + eps/2, ufl.conditional(d < 0, eps, 
                            ((d)**3)/(eps**2) -((d)**4)/(2*eps**3) + eps))

def regularize( nrm, degree = 2 , t: float = 1.0e-5, gamma:float= 0.5 ): 
    A = 0.25*t**(-3/2 - (degree-1)  )    
    C = t**(-0.5) - A*t**degree 
    return  ufl.conditional( ufl.ge(t, nrm), A*nrm**degree + C, nrm**( -1+gamma) )

  
def kappa(h, z, alpha: float = 5/3, gamma:float= 0.5, eps: float = 1e-5 ):
    nrm = ufl.inner( ufl.grad(h+z), ufl.grad(h+z) )**0.5
    return  smooth_cutoff_func(h)**alpha*regularize(nrm, gamma=gamma) 







dt = dolfinx.fem.Constant(mesh, 0.05)
Mn = dolfinx.fem.Constant(mesh, 0.02) 

f = dolfinx.fem.Constant(mesh, 0.02)


u, v = ufl.TrialFunction(V), ufl.TestFunction(V)

h  = dolfinx.fem.Function(V)

hn  = dolfinx.fem.Function(V)

h.x.array[:] = 1.0

z = dolfinx.fem.Function(V)
x_ = ufl.SpatialCoordinate(mesh)
elevation =  dolfinx.fem.Expression( ( 1+ x_[1])*ufl.exp( 2*x_[0]), V.element.interpolation_points() , comm=mesh.comm)
z.interpolate(elevation) 




#Quadrature rule for edge  
q_facet, q_facetw = basix.quadrature.make_quadrature(basix.CellType.interval, 2, basix.quadrature.QuadratureType.default)

#Quadrature rule for triangle 
q_cell, q_cellw = basix.quadrature.make_quadrature(basix.CellType.triangle, 2, basix.quadrature.QuadratureType.default) 

# Quadrature rule for the jacobian.
q_jaco, q_jacow = basix.quadrature.make_quadrature(basix.CellType.triangle, 1, basix.quadrature.QuadratureType.default)


# Assemble F 
#F = ufl.inv(Mn)*dt*ufl.dot(kappa(z,h)*ufl.grad(h+z),ufl.grad(v))*ufl.dx +  h*v*ufl.dx - hn*v*ufl.dx - dt*f*h*ufl.dx 
#Assemble J
#J =  u*v*ufl.dx + ufl.inv(Mn)*dt*ufl.inner( (5/3)*u*kappa(z,h,2/3)*ufl.grad(h+z),ufl.grad(v) ) + ufl.inv(Mn)*dt*ufl.inner( (1/2)*u*kappa(z,h)*ufl.grad(u),ufl.grad(v) ) 


#expr_test =  dolfinx.fem.Expression( dolfinx.fem.Constant(mesh, 1.0) , q_facet, comm=mesh.comm)
#print( expr_test.eval( mesh, np.arange(num_cells, dtype=np.int32)))



a = dolfinx.fem.form(ufl.inner(u, v) * ufl.dx)
A0 = assemble_matrix(a)
A0.assemble()
A0.zeroEntries()


L = dolfinx.fem.form(ufl.inner(hn, v) * ufl.dx  + ufl.inner(f, v) * ufl.dx )
b =dolfinx.fem.petsc.assemble_vector(L)
b.assemble()



"""
   h  := # water height
   z  := # elevation of terrain
   H  := # total elevatio, i.e. h+z 

   For J : 

        [ (5/3)*h^{2/3}/||grad(H)||^{1/2} grad(H)*u +(1/2)*h^{5/3}/||grad(H)||^{1/2} grad(u) ] grad(v)  

        beta = (5/3)*h^{2/3}/||grad(H)||^{1/2} grad(H)

        alpha = (1/2)*h^{5/3}/||grad(H)||^{1/2} 
        
        psi : = (10/3)*grad(H)/ h , 
        
        
   
   For F :       
        
        h^{5/3}/||grad(H)||^{1/2} grad(H) with H = h + z 
        
        -> [ h^{2/3}/||grad(H)||^{1/2}grad(z)*h + h^{5/3}/||grad(H)||^{1/2}grad(h)] grad(v) 

        beta = h^{2/3}/||grad(H)||^{1/2}grad(z)

        alpha = h^{5/3}/||grad(H)||^{1/2}
        
        psi : = grad(z)/ h
        
"""




T_j = A0.copy()

T_f = A0.copy()

alpha_expr = dolfinx.fem.Expression( dt*0.5*ufl.inv(Mn)*dt*kappa(z,h) ,  q_jaco, comm=mesh.comm ) 
alpha =  alpha_expr.eval(mesh, np.arange(num_cells, dtype=np.int32)).reshape(num_cells)

psi_expr = dolfinx.fem.Expression( (10/3)*ufl.grad(h+z)/smooth_cutoff_func(h) ,  q_jaco, comm=mesh.comm ) 
psi =  psi_expr.eval(mesh, np.arange(num_cells, dtype=np.int32)).reshape(num_cells,2)

alpha_f_expr = dolfinx.fem.Expression( dt*ufl.inv(Mn)*dt*kappa(z,h) ,  q_jaco, comm=mesh.comm ) 
alpha_f =  alpha_expr.eval(mesh, np.arange(num_cells, dtype=np.int32)).reshape(num_cells)

psi_f_expr = dolfinx.fem.Expression(  ufl.grad(z)/smooth_cutoff_func(h) ,  q_jaco, comm=mesh.comm ) 
psi_f =  psi_expr.eval(mesh, np.arange(num_cells, dtype=np.int32)).reshape(num_cells,2)



#assemble_eafe( A_global, x, dofmap, phi, grad_phi, detJ, cellwise_alpha, cellwise_psi, mode ): 

trial_expr     = dolfinx.fem.Expression(u, q_cell, comm=mesh.comm)
gradtrial_expr = dolfinx.fem.Expression(ufl.grad(u), q_cell, comm=mesh.comm)
detJ_expr      = dolfinx.fem.Expression(ufl.det(ufl.Jacobian(mesh)), q_jaco, comm=mesh.comm)

phi  = trial_expr.eval(mesh, np.arange(num_cells, dtype=np.int32) ).reshape(num_cells,3,3) 
gphi = gradtrial_expr.eval(mesh, np.arange(num_cells, dtype=np.int32)).reshape(num_cells,6,3)
detJ = detJ_expr.eval(mesh, np.arange(num_cells, dtype=np.int32) )



# New functions -> assemble_dw_rhs() , assemble_dw_lhs() 

assemble_dw_lhs(T_j.handle, x, unrolled_map, phi, gphi, detJ, alpha, psi, PETSc.InsertMode.ADD_VALUES)   
T_j.assemble()

T_j.convert("dense")
neatprint( T_j.getDenseArray() ) 


# Implement dw_rhs
assemble_dw_lhs(T_f.handle, x, unrolled_map, phi, gphi, detJ, alpha_f, psi_f, PETSc.InsertMode.ADD_VALUES)   
T_f.assemble()
 
# dolfinx.la.create_petsc_vector(Vh.dofmap.index_map, Vh.dofmap.index_map_bs) 

# Better method 
h_petsc = assemble_vector( dolfinx.fem.form( h*v*ufl.dx )) 
 
b_f  = T_f*h_petsc  

b += b_f

"""
    J(u,v;h)dh = F(v;h) 
    
    h +=dh
    
    J(u,v;h) = A_j + T_j
    
    F(v;h)   = (A_f + T_f)*h + b 

    b = - (h_n,v) - dt*(f,v)  

"""

sol = T_j.createVecRight()
ksp = PETSc.KSP().create()
ksp.setOperators(T_j)
ksp.setType('preonly')
ksp.setConvergenceHistory()
ksp.getPC().setType('lu')

ksp.solve(b, sol )

print( sol.norm())





exit()

import matplotlib.pyplot as plt 

plt.semilogy(ksp.getConvergenceHistory())

plt.show()

"""

    
    
    F(v;h) = (h,v) + dt*h^{5/3}/||grad(H)||^{1/2}( grad(H), grad(v) ) - (hn,v) - dt*(f,v) - dt*h^{5/3}/||grad(H)||^{1/2}( grad(H)*n,v)
  
    2D - 1D coupling :     
    
                   s(v;h)   =   dt*h^{5/3}/||grad(H)||^{1/2}( grad(H)*n,v)
                    
                   eafe :  
                            beta = h^{2/3}/||grad(H)||^{1/2}grad(z)

                            alpha = h^{5/3}/||grad(H)||^{1/2}
        
                            psi : = grad(z)/ h

                   ==>   s(v;h)  = [ dt*alpha*B(psi*n)*v ] *h 
                    


                   S(u,v;h)  = ([dt*(5/3)*h^{2/3}/||grad(H)||^{1/2} grad(H)*u +(1/2)*h^{5/3}/||grad(H)||^{1/2} grad(u)],n*v)                 
                    
                   eafe :  
                            beta = h^{2/3}/||grad(H)||^{1/2}grad(z)
                    
                            alpha = h^{5/3}/||grad(H)||^{1/2}
        
                            psi : = grad(z)/ h

                   ==>   S(u,v;h)  = dt*alpha*B(psi*n)*v      

   1D has a width, 
   
   
   Step 1 : 
   
             Assemble system for 1D : 
                                     - find all marked edges and map to dofs 
                                     - implement aefe 
                                     - assemble system   
             
             
                                      
   Step 2 : 
             Assemble coupling term 2D-1D 
                                     - duplicate dofs 
                                     - implemente eafe 
                                     - assemble system 
             
             
             
   Step 3 : 
             Assemble coupling term 1D-2D           
                       
   
"""




















