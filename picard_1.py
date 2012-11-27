"""
The non-linear diffusion equation

rho*du/dt = \nabla \dot (alpha(u)\nabla u) + f(x, t),

with x \in Omega, t \in (0, T], is solved with only
one Picard iteration at every time step.

The PDE is discretized in time using the Backward 
Euler method. 

"""
import scitools.std as sci
import nose.tools as nt
from dolfin import *
import numpy, sys

Ct = 1.0
Cx = 0.01

h_values = 0.05, 0.01, 0.005, 0.001, 0.0005, 0.0001, 0.00001
errors = []

for h in h_values:

    #     degree = degree of finite element approximation
    #     n_elements = list of number of elements
    #                  for each dimension
    #     dim = physical space dimension
    #     ref_domain = type of reference domain
    #     V = test function space
    #degree = int(sys.argv[1])
    #n_elements = [int(arg) for arg in sys.argv[2:]]
    degree = 1
    #n_elements = [1, 1]
    Nx = int(round(1.0/numpy.sqrt(Cx*h)))
    n_elements = [Nx]
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
    #print 'u_1 = ', u_1.vector().array()
    
    #     dt = time step length
    #     T = stopping time for simulation
    dt = Ct*h
    T = 0.05
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
    
    f = open('x_u.dat','w')
    fe = open('x_ue.dat','w')
    
    u = Function(V)
    t = dt
    #     Loop over time
    while t <= T:
        #     Assemble the matrix and vector
        A = assemble(a)
        b = assemble(L)
        #     Solve the linear system of equations
        solve(A, u.vector(), b)
        
        #print '>> ', 1.0/3 + dt, ', ', 1.0/6 - dt
        #print '>> ', (1.0/3 + dt)**2, ', ', (1.0/6 - dt)**2
        #print ':: ', 0.89285714/3 - 0.89285714/6
        #print ':: ', 0.89285714/6 - 0.89285714/3
        #print 'A = ', A.array()
        #print 'b = ', b.array()
        #print 'c = ', u.vector().array()
        #plot(u)
        #plot(mesh)
        
        #     Test the result against the exact solution
        u0.t = t
        u_e = interpolate(u0, V)
        #print 'u_e=',u_e.vector().array()
        #print 'exp(-pi**2*t)=',exp(-pi**2*t)
        e = u_e.vector().array() - u.vector().array()
        E = numpy.sqrt(numpy.sum(e**2)/u.vector().array().size)
        
        #print 'h=',h,',t=',t
        if t+dt > T: 
            errors.append(E)
            print 'appended'

        #print 'Error = ', E, ', e_max = ', e.max()
        
        #print 'u_1 = ', u.vector().array()
        #print 'mesh=',mesh.coordinates()
        
        uv = u.vector().array()
        x = numpy.zeros(len(uv))
        for i in range(len(uv)): 
            x[i] = mesh.coordinates()[i][0]
            if t > T-dt:
                f.write(str(mesh.coordinates()[i][0])
                        +' '+str(u.vector().array()[i])+'\n')
                fe.write(str(mesh.coordinates()[i][0])
                         +' '+str(u_e.vector().array()[i])+'\n')
    
        #print 'x=',x
        #sci.plot(uv, uv)
    
        #interactive()
    
        t += dt
        u_1.assign(u)

print 'errors = ', errors

nh = len(h_values)
        
#     Calculate convergence rates
r = [numpy.log(errors[i-1]/errors[i])/ \
         numpy.log(h_values[i-1]/h_values[i]) \
         for i in range(1, nh, 1)]

for i in range(1, nh):
    print h_values[i-1], r[i-1]

#diff = abs(r[nh-2]-1.0)
#print 'diff = ', diff
#nt.assert_almost_equal(diff, 0, delta=1E-1)
    
f.close()
fe.close()
