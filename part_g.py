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

Ct = 0.1
Cx = 0.01

#h_values = [10.0, 5.0, 2.0, 1.0, 0.5, 0.01]#, 0.005, 0.001
h_values = [0.1, 0.05, 0.02, 0.01, 0.005]
errors = []

file_1 = open('x_u.dat','w')
file_e = open('x_ue.dat','w')

for h in h_values:
    
    #     degree = degree of finite element approximation
    #     n_elements = list of number of elements
    #                  for each dimension
    #     dim = physical space dimension
    #     ref_domain = type of reference domain
    #     V = test function space
    degree = 1
    Nx = int(round(1.0/sqrt(Cx*h)))
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
    #u0 = Expression('exp(-pi*pi*t)*cos(pi*x[0])', t=0.0)
    u0 = Expression('x[0]*x[0]*(0.5 - x[0]/3.0)*t', t=0.0)
    #u0 = Expression('t', t=0.0)
    u_1 = interpolate(u0, V)
    u_q1 = interpolate(u0, V)
    
    #     dt = time step length
    #     T = stopping time for simulation
    T = 1.0
    Nt = int(round(T/(Ct*h)))
    dt = T/Nt
    print 'Nx = ', Nx, ', dt = ', dt
    
    #     Constant rho
    rho = 1.0
    
    #f = Constant(1.0)
    #f = Constant(0.0)
    #f = Expression('-x[0]*x[0]*x[0]/3.0 + 0.5*x[0]*x[0]'\
    #                   ' + 2.0*t*x[0] - t', t=0.0)
    
    #     Source function determined by the manufactured
    #     solution u0 and the diffusion coefficient 
    #     alpha.
    f = Expression('-rho*pow(x[0], 3)/3.0 + rho*x[0]*x[0]/2.0 '\
                       '+ 8.0*pow(t, 3)*pow(x[0], 7)/9.0 '\
                       '- 28.0*pow(t, 3)*pow(x[0], 6)/9.0 '\
                       '+ 7.0*pow(t, 3)*pow(x[0], 5)/2.0 '\
                       '- 5.0*pow(t, 3)*pow(x[0], 4)/4.0 '\
                       '+ 2*t*x[0] - t', t=0.0, rho=rho)

    # f = Expression('rho*x[0]*x[0]*(-2*x[0] + 3)/6 -'\
    #                   '(-12*t*x[0] + 3*t*(-2*x[0] + 3))'\
    #                   '*(pow(x[0], 4)*pow((-dt + t), 2)'\
    #                   '*pow((-2*x[0] + 3), 2) + 36)/324'\
    #                   '- (-6*t*x[0]*x[0] + 6*t*x[0]*(-2*x[0] + 3))'\
    #                   '*(36*pow(x[0], 4)*pow((-dt + t), 2)*(2*x[0] - 3)'\
    #                   '+ 36*pow(x[0], 3)*pow((-dt + t), 2)'\
    #                   '*pow((-2*x[0] + 3), 2))'\
    #                   '/5832', t=0.0, dt=dt, rho=rho)
    
    # f = Expression('8*dt*dt*t*pow(x[0], 7)/9 '\
    #                    '- 28*dt*dt*t*pow(x[0], 6)/9 '\
    #                    '+ 7*dt*dt*t*pow(x[0], 5)/2 '\
    #                    '- 5*dt*dt*t*pow(x[0], 4)/4 '\
    #                    '- 16*dt*t*t*pow(x[0], 7)/9 '\
    #                    '+ 56*dt*t*t*pow(x[0], 6)/9 '\
    #                    '- 7*dt*t*t*pow(x[0], 5) '\
    #                    '+ 5*dt*t*t*pow(x[0], 4)/2 '\
    #                    '- rho*pow(x[0], 3)/3 + rho*x[0]*x[0]/2 '\
    #                    '+ 8*pow(t, 3)*pow(x[0], 7)/9 '\
    #                    '- 28*pow(t, 3)*pow(x[0], 6)/9 '\
    #                    '+ 7*pow(t, 3)*pow(x[0], 5)/2 '\
    #                    '- 5*pow(t, 3)*pow(x[0], 4)/4'\
    #                    '+ 2*t*x[0] - t', t=0.0, dt=dt, rho=rho)
    
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
    max_iter = 30
    #     Loop over time
    while t <= T:
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
                             0.00000001)
            error = norm_diff/norm_u
            #print 'rel_diff=',error
            #     Update u_q1
            u_q1.assign(u)
        
        #print 'iter = ', iteration
        #print 'rel error = ', error
        
        #     Test the converged result against the 
        #     exact manufactured solution
        u0.t = t
        u_e = interpolate(u0, V)
        e = u_e.vector().array() - u.vector().array()
        E = sqrt(sum(e**2)/u.vector().array().size)
        
        #print 'error = ', E
        if t+dt > T: 
            errors.append(E)
            #print 't+dt=',t+dt,',T=',T
    
            uv = u.vector().array()
            x = zeros(len(uv))
            for i in range(len(uv)): 
                file_1.write(str(mesh.coordinates()[i][0])
                             +' '+str(u.vector().array()[i])+'\n')
                file_e.write(str(mesh.coordinates()[i][0])
                             +' '+str(u_e.vector().array()[i])+'\n')
            
        t += dt
        u_1.assign(u)
        
print 'errors = ', errors
nh = len(h_values)

#     Calculate convergence rates
r = [log(errors[i-1]/errors[i])/ \
         log(h_values[i-1]/h_values[i]) \
         for i in range(1, nh, 1)]
#     Print convergence parameter h and convergence
#     rate r
for i in range(1, nh):
    print h_values[i-1], r[i-1]

#     Check that convergence rate is 1 with a nose
#     test
#diff = abs(r[nh-2]-1.0)
#print 'diff = ', diff
#nt.assert_almost_equal(diff, 0, delta=1E-2)

file_1.close()
file_e.close()
