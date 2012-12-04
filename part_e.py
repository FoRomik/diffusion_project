"""
The non-linear diffusion equation

rho*du/dt = \nabla \dot (alpha(u)\nabla u) + f(x, t),

with x \in Omega, t \in (0, T], is solved with only
one Picard iteration at every time step.

The PDE is discretized in time using the Backward 
Euler method. 

"""

import nose.tools as nt
from dolfin import *
from numpy import *
from sys import *
    
#     degree = degree of finite element approximation
#     n_elements = list of number of elements
#                  for each dimension
#     dim = physical space dimension
#     ref_domain = type of reference domain
#     V = test function space
degree = 1
n_elements = [1000]
dim = len(n_elements)
ref_domain = [UnitInterval, UnitSquare, UnitCube]
mesh = ref_domain[dim-1](*n_elements)
V = FunctionSpace(mesh, 'Lagrange', degree)

#     The Neumann boundary conditions are 
#     implicitly included in the variational
#     formulation.     
#
#     Initial condition
u0 = Expression('x[0]*0.0')
u_1 = interpolate(u0, V)
u_q1 = interpolate(u0, V)
#     Manufactured solution
u_mf = Expression('x[0]*x[0]*(0.5 - x[0]/3.0)*t', t=0.0)
    
#     dt = time step length
#     T = stopping time for simulation
dt = 0.00000001
T = 0.00001

#     Constant rho
rho = 1.0

#     Source function determined by the manufactured
#     solution u_mf and the diffusion coefficient 
#     alpha.
f = Expression('-rho*pow(x[0], 3)/3 + rho*x[0]*x[0]/2 '\
                   '+ 8*pow(t, 3)*pow(x[0], 7)/9 '\
                   '- 28*pow(t, 3)*pow(x[0], 6)/9 '\
                   '+ 7*pow(t, 3)*pow(x[0], 5)/2 '\
                   '- 5*pow(t, 3)*pow(x[0], 4)/4 '\
                   '+ 2*t*x[0] - t', t=0.0, rho=rho)

#     Diffusion coefficient
def alpha(u):
    return 1.0 + u**2

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
#     Loop over time
while t <= T:
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
        #print 'rel_diff=',error
        #     Update u_q1
        u_q1.assign(u)
    
    #print 'iter = ', iteration
    #     Test the converged result against the 
    #     exact manufactured solution
    u_mf.t = t
    u_e = interpolate(u_mf, V)
    e = u_e.vector().array() - u.vector().array()
    E = sqrt(sum(e**2)/u.vector().array().size)
    
    #print 'error = ', E
    
    t += dt
    u_1.assign(u)

print 'E = ', E
