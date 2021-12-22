import os
import sys

import crocoddyl
import pinocchio
import numpy as np
import example_robot_data

import libipopt_croco_pywrap as ipcroco

talos_arm = example_robot_data.load('talos_arm')
robot_model = talos_arm.model

state = crocoddyl.StateMultibody(robot_model)
runningCostModel = crocoddyl.CostModelSum(state)
terminalCostModel = crocoddyl.CostModelSum(state)

framePlacementResidual = crocoddyl.ResidualModelFramePlacement(
    state, robot_model.getFrameId("gripper_left_joint"),
    pinocchio.SE3(np.eye(3), np.array([.0, .0, -1])))
uResidual = crocoddyl.ResidualModelControl(state)
xResidual = crocoddyl.ResidualModelControl(state)
goalTrackingCost = crocoddyl.CostModelResidual(state, framePlacementResidual)
xRegCost = crocoddyl.CostModelResidual(state, xResidual)
uRegCost = crocoddyl.CostModelResidual(state, uResidual)

runningCostModel.addCost("gripperPose", goalTrackingCost, 1)
runningCostModel.addCost("xReg", xRegCost, 1e-4)
runningCostModel.addCost("uReg", uRegCost, 1e-1)
terminalCostModel.addCost("gripperPose", goalTrackingCost, 1)

actuationModel = crocoddyl.ActuationModelFull(state)
dt = 1e-2
runningModel = crocoddyl.IntegratedActionModelEuler(
    crocoddyl.DifferentialActionModelFreeFwdDynamics(state, actuationModel,
                                                     runningCostModel), dt)
runningModel.differential.armature = np.array(
    [0.1, 0.1, 0.1, 0.1, 0.1, 0.1, 0.])
terminalModel = crocoddyl.IntegratedActionModelEuler(
    crocoddyl.DifferentialActionModelFreeFwdDynamics(state, actuationModel,
                                                     terminalCostModel), 0.)
terminalModel.differential.armature = np.array(
    [0.1, 0.1, 0.1, 0.1, 0.1, 0.1, 0.])

T = 100
q0 = np.array([0.173046, 1., -0.52366, 0., 0., 0.1, -0.005])
x0 = np.concatenate([q0, pinocchio.utils.zero(robot_model.nv)])
problem = crocoddyl.ShootingProblem(x0, [runningModel] * T, terminalModel)

ddp = crocoddyl.SolverFDDP(problem)
ddp.setCallbacks([crocoddyl.CallbackVerbose()])

ddp.solve()

solver = ipcroco.SolverIpOpt(problem)
solver.solve()

display = crocoddyl.GepettoDisplay(talos_arm)
display.display(solver.xs, dts=[dt] * len(solver.xs), factor=4)

pass
