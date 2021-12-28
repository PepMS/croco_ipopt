import os
import sys

import crocoddyl
import pinocchio
import numpy as np
import example_robot_data

import matplotlib.pyplot as plt

import libipopt_croco_pywrap as ipcroco

WITHDISPLAY = 'display' in sys.argv or 'CROCODDYL_DISPLAY' in os.environ
WITHPLOT = 'plot' in sys.argv or 'CROCODDYL_PLOT' in os.environ

hector = example_robot_data.load('hector')
robot_model = hector.model

target_pos = np.array([1., 0., 1.])
target_quat = pinocchio.Quaternion(1., 0., 0., 0.)

state = crocoddyl.StateMultibody(robot_model)

d_cog, cf, cm, u_lim, l_lim = 0.1525, 6.6e-5, 1e-6, 5., 0.1
tau_f = np.array([[0., 0., 0., 0.], [0., 0., 0., 0.], [1., 1., 1., 1.],
                  [0., d_cog, 0., -d_cog], [-d_cog, 0., d_cog, 0.],
                  [-cm / cf, cm / cf, -cm / cf, cm / cf]])
actuation = crocoddyl.ActuationModelMultiCopterBase(state, tau_f)

nu = actuation.nu
runningCostModel = crocoddyl.CostModelSum(state, nu)
terminalCostModel = crocoddyl.CostModelSum(state, nu)

# Costs
xResidual = crocoddyl.ResidualModelState(state, state.zero(), nu)
xActivation = crocoddyl.ActivationModelWeightedQuad(
    np.array([1.] * 3 + [1.] * 3 + [1.] * robot_model.nv))
uResidual = crocoddyl.ResidualModelControl(state, nu)
xRegCost = crocoddyl.CostModelResidual(state, xActivation, xResidual)
uRegCost = crocoddyl.CostModelResidual(state, uResidual)
goalTrackingResidual = crocoddyl.ResidualModelFramePlacement(
    state, robot_model.getFrameId("base_link"),
    pinocchio.SE3(target_quat.matrix(), target_pos), nu)
goalTrackingCost = crocoddyl.CostModelResidual(state, goalTrackingResidual)
runningCostModel.addCost("xReg", xRegCost, 1e-3)
runningCostModel.addCost("uReg", uRegCost, 1e-3)
terminalCostModel.addCost("goalPose", goalTrackingCost, 3.)

dt = 1e-2
runningModel = crocoddyl.IntegratedActionModelEuler(
    crocoddyl.DifferentialActionModelFreeFwdDynamics(state, actuation,
                                                     runningCostModel), dt)
terminalModel = crocoddyl.IntegratedActionModelEuler(
    crocoddyl.DifferentialActionModelFreeFwdDynamics(state, actuation,
                                                     terminalCostModel), dt)
runningModel.u_lb = np.array([l_lim, l_lim, l_lim, l_lim])
runningModel.u_ub = np.array([u_lim, u_lim, u_lim, u_lim])

# Creating the shooting problem and the BoxDDP solver
T = 100
problem = crocoddyl.ShootingProblem(
    np.concatenate([hector.q0, np.zeros(state.nv)]), [runningModel] * T,
    terminalModel)

fddp = crocoddyl.SolverBoxDDP(problem)
fddp.setCallbacks([crocoddyl.CallbackVerbose()])
fddp.solve([], [], 200)

solver = ipcroco.SolverIpOpt(problem)
solver.solve(fddp.xs, fddp.us, 500)

fig, axs = plt.subplots(nrows=4)

for i in range(4):
    axs[i].plot([u[i] for u in solver.us])

fig2, axs2 = plt.subplots(nrows=13)

for i in range(13):
    axs2[i].plot([x[i] for x in solver.xs])

plt.show()