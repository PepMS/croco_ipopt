import os
import sys

import crocoddyl
import numpy as np
import example_robot_data

from crocoddyl.utils.pendulum import CostModelDoublePendulum, ActuationModelDoublePendulum

import libipopt_croco_pywrap as ipcroco

# Loading the double pendulum model
pendulum = example_robot_data.load('double_pendulum')
model = pendulum.model

state = crocoddyl.StateMultibody(model)
actuation = ActuationModelDoublePendulum(state, actLink=1)

nu = actuation.nu
runningCostModel = crocoddyl.CostModelSum(state, nu)
terminalCostModel = crocoddyl.CostModelSum(state, nu)

xResidual = crocoddyl.ResidualModelState(state, state.zero(), nu)
xActivation = crocoddyl.ActivationModelQuad(state.ndx)
uResidual = crocoddyl.ResidualModelControl(state, nu)
xRegCost = crocoddyl.CostModelResidual(state, xActivation, xResidual)
uRegCost = crocoddyl.CostModelResidual(state, uResidual)
xPendCost = CostModelDoublePendulum(
    state,
    crocoddyl.ActivationModelWeightedQuad(np.array([1.] * 4 + [0.1] * 2)), nu)

dt = 1e-2

runningCostModel.addCost("uReg", uRegCost, 1e-4 / dt)
runningCostModel.addCost("xGoal", xPendCost, 1e-5 / dt)
terminalCostModel.addCost("xGoal", xPendCost, 100.)

runningModel = crocoddyl.IntegratedActionModelEuler(
    crocoddyl.DifferentialActionModelFreeFwdDynamics(state, actuation,
                                                     runningCostModel), dt)
terminalModel = crocoddyl.IntegratedActionModelEuler(
    crocoddyl.DifferentialActionModelFreeFwdDynamics(state, actuation,
                                                     terminalCostModel), dt)

# Creating the shooting problem and the FDDP solver
T = 100
x0 = np.array([3.14, 0., 0., 0.])
problem = crocoddyl.ShootingProblem(x0, [runningModel] * T, terminalModel)
problem.nthreads = 1  # TODO(cmastalli): Remove after Crocoddyl supports multithreading with Python-derived models
fddp = crocoddyl.SolverFDDP(problem)

fddp.solve()

solver = ipcroco.SolverIpOpt(problem)
solver.solve()

pass