import os
import sys

import crocoddyl
import pinocchio
import numpy as np
import example_robot_data

import libipopt_croco_pywrap as ipcroco


WITHDISPLAY = 'display' in sys.argv or 'CROCODDYL_DISPLAY' in os.environ
WITHPLOT = 'plot' in sys.argv or 'CROCODDYL_PLOT' in os.environ

hector = example_robot_data.load('hector')
robot_model = hector.model

target_pos = np.array([1., 0., 1.])
target_quat = pinocchio.Quaternion(0., 0., 0., 1.)
target_quat.normalize()

state = crocoddyl.StateMultibody(robot_model)

d_cog, cf, cm, u_lim, l_lim = 0.1525, 6.6e-5, 1e-6, 5., 0.1
tau_f = np.array([[0., 0., 0., 0.], [0., 0., 0., 0.], [1., 1., 1., 1.], [0., d_cog, 0., -d_cog],
                  [-d_cog, 0., d_cog, 0.], [-cm / cf, cm / cf, -cm / cf, cm / cf]])
actuation = crocoddyl.ActuationModelMultiCopterBase(state, tau_f)

nu = actuation.nu
runningCostModel = crocoddyl.CostModelSum(state, nu)
terminalCostModel = crocoddyl.CostModelSum(state, nu)

# Costs
xResidual = crocoddyl.ResidualModelState(state, state.zero(), nu)
xActivation = crocoddyl.ActivationModelWeightedQuad(np.array([0.1] * 3 + [1000.] * 3 + [1000.] * robot_model.nv))
uResidual = crocoddyl.ResidualModelControl(state, nu)
xRegCost = crocoddyl.CostModelResidual(state, xActivation, xResidual)
uRegCost = crocoddyl.CostModelResidual(state, uResidual)
goalTrackingResidual = crocoddyl.ResidualModelFramePlacement(state, robot_model.getFrameId("base_link"),
                                                             pinocchio.SE3(target_quat.matrix(), target_pos), nu)
goalTrackingCost = crocoddyl.CostModelResidual(state, goalTrackingResidual)
runningCostModel.addCost("xReg", xRegCost, 1e-3)
runningCostModel.addCost("uReg", uRegCost, 1e-2)
# runningCostModel.addCost("trackPose", goalTrackingCost, 1e-2)
terminalCostModel.addCost("goalPose", goalTrackingCost, 100.)

dt = 1e-2
runningModel = crocoddyl.IntegratedActionModelEuler(
    crocoddyl.DifferentialActionModelFreeFwdDynamics(state, actuation, runningCostModel), dt)
terminalModel = crocoddyl.IntegratedActionModelEuler(
    crocoddyl.DifferentialActionModelFreeFwdDynamics(state, actuation, terminalCostModel), dt)

# Creating the shooting problem and the FDDP solver
T = 30
problem = crocoddyl.ShootingProblem(np.concatenate([hector.q0, np.zeros(state.nv)]), [runningModel] * T, terminalModel)
fddp = crocoddyl.SolverFDDP(problem)

fddp.setCallbacks([crocoddyl.CallbackLogger(), crocoddyl.CallbackVerbose()])

fddp.solve()

solver = ipcroco.SolverIpOpt(problem)
# solver.solve(fddp.xs, fddp.us, 1000)
solver.solve()

display = crocoddyl.GepettoDisplay(hector)
display.display(solver.xs, dts=[dt] * len(solver.xs), factor=4)

pass