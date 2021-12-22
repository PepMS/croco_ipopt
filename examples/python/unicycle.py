import numpy as np
import crocoddyl
import libipopt_croco_pywrap as ipcroco

model = crocoddyl.ActionModelUnicycle()

# Setting up the cost weights
N = 20
model.r = [10., 1.]
x0 = np.array([-1., -1., 1.]).T  #x,y,theta


problem = crocoddyl.ShootingProblem(x0, [model] * N, model)

solver = ipcroco.SolverIpOpt(problem)

solver.solve()

print()