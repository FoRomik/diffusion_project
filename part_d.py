"""
The non-linear diffusion equation

rho*du/dt = \nabla \dot (alpha(u)\nabla u) + f(x, t),

with x \in Omega, t \in (0, T], is solved with only
one Picard iteration at every time step.

The PDE is discretized in time using the Backward 
Euler method. 

In this file we test the convergence rate of a
simple linear problem.

"""
import scitools.std as sci
import nose.tools as nt
from dolfin import *
import numpy, sys

Ct = 1.0
Cx = 0.01

h_values = [0.05, 0.01, 0.005, 0.001]
errors = []

file_1 = open('x_u_d3.dat','w')
file_e = open('x_ue_d3.dat','w')

for h in h_values:
    
    #     degree:        Degree of finite element 
    #                    approximation
    #     n_elements:    List of number of elements
    #                    for each dimension
    #     dim:           Physical space dimension
    #     ref_domain:    Type of reference domain
    #     V:             Test function space
    degree = 1
    Nx = int(round(1.0/numpy.sqrt(Cx*h)))
    n_elements = [Nx, Nx]
    dim = len(n_elements)
    ref_domain = [UnitInterval, UnitSquare, UnitCube]
    mesh = ref_domain[dim-1](*n_elements)
    V = FunctionSpace(mesh, 'Lagrange', degree)
    
    #     The Neumann boundary conditions are 
    #     implicitly included in the variational
    #     formulation.     
    #
    #     Initial condition
    u0 = Expression('exp(-pi*pi*t)*cos(pi*x[0])', t=0.0)
    u_1 = interpolate(u0, V)
    
    #     dt:       Time step length
    #     T:        Stopping time for simulation
    T = 0.05
    Nt = int(round(T/(Ct*h)))
    dt = T/Nt
    print 'Nx = ', Nx, ', dt = ', dt
    
    #     Diffusion coefficient
    def alpha(u):
        return 1.0
    
    #     Source function
    f = Constant(0.0)
    #     Constant rho
    rho = 1.0
    
    #     Set up the variational formulation
    u = TrialFunction(V)
    v = TestFunction(V)
    a = (rho*inner(u, v) \
             + dt*inner(alpha(u_1)*nabla_grad(u), \
                            nabla_grad(v)))*dx
    L = (dt*inner(f, v) + rho*inner(u_1, v))*dx
    
    u = Function(V)
    t = dt
    #     Loop over time
    while t <= T:
        f.t = t
        #     Assemble the matrix and vector
        A = assemble(a)
        b = assemble(L)
        #     Solve the linear system of equations
        solve(A, u.vector(), b)
        
        #     Test the result against the exact solution
        u0.t = t
        u_e = interpolate(u0, V)
        e = u_e.vector().array() - u.vector().array()
        E = numpy.sqrt(numpy.sum(e**2) \
                           /u.vector().array().size)
        
        if t+dt > T: 
            errors.append(E)
            #plot(u)
            #interactive()

            #     Write numerical and exact solutions
            #     to file
            # uv = u.vector().array()
            # x = numpy.zeros(len(uv))
            # for i in range(len(uv)): 
            #     file_1.write(str(mesh.coordinates()[i][0])
            #                  +' '+str(u.vector().array()[i])+'\n')
            #     file_e.write(str(mesh.coordinates()[i][0])
            #                  +' '+str(u_e.vector().array()[i])+'\n')

        t += dt
        u_1.assign(u)

print 'errors = ', errors
nh = len(h_values)
        
#     Calculate convergence rates
r = [numpy.log(errors[i-1]/errors[i])/ \
         numpy.log(h_values[i-1]/h_values[i]) \
         for i in range(1, nh, 1)]
#     Print convergence parameter h and convergence
#     rate r
for i in range(1, nh):
    print h_values[i-1], r[i-1]

#     Check that convergence rate is 1 with a nose
#     test
diff = abs(r[nh-2]-1.0)
print 'diff = ', diff
nt.assert_almost_equal(diff, 0, delta=1E-2)

file_1.close()
file_e.close()
