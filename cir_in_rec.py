from firedrake import * 
import matplotlib.pyplot as plt
# parameters
T = 0.2
dt = Constant(0.000001)
t = Constant(0)

base = Mesh("mesh/circle_in_rect.msh")
mesh = Submesh(base, 2, 2)
(x, y)= SpatialCoordinate(mesh)

V = mesh.coordinates.function_space()
X = Function(V)
Xp = Function(V)
v = TestFunction(V)
Xp.interpolate(mesh.coordinates)

F = (
    inner((X - Xp)/dt, v) * dx
    + inner(grad(X), grad(v)) * dx
)
pb = NonlinearVariationalProblem(F, X)
solver = NonlinearVariationalSolver(pb, solver_parameters={"ksp_type": "preonly", "pc_type":"lu"})
pvd = VTKFile("output/results.pvd")
# time loop
plt.ion()   
fig = plt.figure(figsize=(6,6))
ax = fig.add_subplot(111)
ax.set_aspect("equal")
while (float(t) < float(T-dt) + 1.0e-10):
    t.assign(t+dt)    
    if mesh.comm.rank==0:
        print(f"Solving for t = {float(t):.4f} .. ", flush=True)
    solver.solve()
    mesh.coordinates.assign(X)
    ax.clear()   
    ax.set_aspect("equal")
    triplot(mesh)           
    ax.set_title(f"t = {float(t):.4f}")
    fig.canvas.draw()
    plt.pause(0.2)       
    pvd.write(X, time=float(t))
    Xp.assign(X)
    
plt.ioff()
plt.show() 
