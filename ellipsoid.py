from firedrake import *
import numpy as np

# parameters
T = 1.0
dt = Constant(0.01)
t = Constant(0)

# mesh
mesh = UnitIcosahedralSphereMesh(2)   # UnitDiskMesh 在 Firedrake 有实现
(x, y, z)= SpatialCoordinate(mesh)
mesh.coordinates.interpolate(as_vector([x, y, 1.5 * z]))

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
while (float(t) < float(T-dt) + 1.0e-10):
    t.assign(t+dt)    
    if mesh.comm.rank==0:
        print(f"Solving for t = {float(t):.4f} .. ", flush=True)
    solver.solve()
    mesh.coordinates.assign(X)
    pvd.write(X, time=float(t))
    Xp.assign(X)
