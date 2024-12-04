from fenics import *
import numpy as np
import matplotlib.pyplot as plt

T = 2.0            # final time
num_steps = 10     # number of time steps
dt = T / num_steps # time step size
alpha = 3          # parameter alpha
beta = 1.2         # parameter beta

# Create mesh and define function space
nx = ny = 8
mesh = UnitSquareMesh(nx, ny)
V = FunctionSpace(mesh, 'P', 1)




alpha = 3; beta = 1.2
u_D = Expression('1 + x[0]*x[0] + alpha*x[1]*x[1] + beta*t',
                 degree=2, alpha=alpha, beta=beta, t=0)

def boundary(x, on_boundary):
    return on_boundary

bc = DirichletBC(V, u_D, boundary)

u_n = interpolate(u_D, V)

u = TrialFunction(V)
v = TestFunction(V)
f = Constant(beta - 2 - 2*alpha)

F = u*v*dx + dt*dot(grad(u), grad(v))*dx - (u_n + dt*f)*v*dx
a, L = lhs(F), rhs(F)

u# Time-stepping
u = Function(V)
t = 0
k = 0
for n in range(num_steps):

    # Update current time
    t += dt
    u_D.t = t # update for bc

    # Compute solution
    solve(a == L, u, bc)

    # Compute error at vertices
    u_e = interpolate(u_D, V)
    error = np.abs(u_e.vector().get_local() - u.vector().get_local()).max()
    print('t = %.2f: error = %.3g' % (t, error))

    plot(u, title=('diffusion time viscous %g' % (t)))
    plt.grid(True)
    filename = ('diff2D_%d.png' % (k))
    plt.savefig(filename)
    print('Graphics saved as "%s"' % (filename))
    plt.close()

    # Update previous solution
    u_n.assign(u)
    k = k + 1
qq = 0