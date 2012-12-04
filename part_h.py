"""
The non-linear diffusion equation

rho*du/dt = \nabla \dot (alpha(u)\nabla u) + f(x, t),

with x \in Omega, t \in (0, T], is solved with only
one Picard iteration at every time step.

The PDE is discretized in time using the Backward 
Euler method. 

Here we have chosen alpha(u) = 1 + beta*u**2, we 
consider a two-dimensional domain x, y \in [0, 1]
x [0, 1], and the initial condition I(x, y) is 
a Gaussian-shaped function.

"""

import nose.tools as nt
from mayavi import mlab
from dolfin import *
from numpy import *
from sys import *
    
#     degree:       Degree of finite element 
#                   approximation
#     n_elements:   List of number of elements
#                   for each dimension
#     dim:          Physical space dimension
#     ref_domain:   Type of reference domain
#     V:            Test function space
degree = 1
n_elements = [20, 20]
dim = len(n_elements)
ref_domain = [UnitInterval, UnitSquare, UnitCube]
mesh = ref_domain[dim-1](*n_elements)
V = FunctionSpace(mesh, 'Lagrange', degree)

#     The Neumann boundary conditions are 
#     implicitly included in the variational
#     formulation.     
#
#     Initial condition
sigma2 = 0.01
u0 = Expression('exp(-0.5*(x[0]*x[0] + x[1]*x[1])/sigma2)', \
                    sigma2=sigma2)
u_1 = interpolate(u0, V)
u_q1 = interpolate(u0, V)
    
#     dt: Time step length
#     T:  Topping time for simulation
dt = 0.01
T = 4.0

#     Constant rho
rho = 1.0

#     Source function determined by the manufactured
#     solution u_mf and the diffusion coefficient 
#     alpha.
f = Expression('exp(-t)*x[1]*sin(10*x[0]*t)', t=0.0)

beta = 100.0
#     Diffusion coefficient
def alpha(u):
    return 1.0 + beta*u**2

#     Set up the variational formulation
u = TrialFunction(V)
v = TestFunction(V)
a = (rho*inner(u, v) \
         + dt*inner(alpha(u_q1)*nabla_grad(u), \
                        nabla_grad(v)))*dx
L = (dt*inner(f, v) + rho*inner(u_1, v))*dx

u = Function(V)
t = dt
tol = 1E-10
max_iter = 10
i_fig = 0
n = 0 
#     Loop over time
while t <= T:
    n = n + 1
    f.t = t
    #     u_q1 = u^{n, q-1}, u_1 = u^{n-1},
    #     where n is the time step and q is
    #     the Picard iteration step.
    u_q1.assign(u_1) 
    
    error = 1.0
    iteration = 0
    #     Picard iteration to solve the nonlinear
    #     equation.
    while error > tol and iteration < max_iter:
        iteration += 1 
        #     Assemble the matrix and vector
        A = assemble(a)
        b = assemble(L)
        #     Solve the linear system of equations.
        solve(A, u.vector(), b)
        
        #     Calculate the relative difference
        #     of u vector 2-norms.
        norm_diff = linalg.norm(u.vector().array() \
                                    - u_q1.vector().array())
        norm_u = max(linalg.norm(u_q1.vector().array()), \
                         0.00001)
        error = norm_diff/norm_u
        
        #     Update u_q1
        u_q1.assign(u)

    #     Write data to files 
    if n%100 == 0:
        print 't = ', t
        i_fig += 1
        plot(u)
        file = File('gaussian_t' + str(t) + '.pvd')
        file << u
        interactive()
        
        fi = interpolate(f, V)
        file = File('f_function_t' + str(t) + '.pvd')
        file << fi 
    
    t += dt
    u_1.assign(u)

