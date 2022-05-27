#==========================================
# Title:  Example for Pinocchio & CasADI
# Author: Pep Martí Saumell
# Date:   27 May 2022
#==========================================

from os.path import join

import casadi
import numpy as np

import pinocchio as pin
import pinocchio.casadi as cpin
from pinocchio.robot_wrapper import RobotWrapper

x_goal = [1, 0, 1.5, 0, 0, 0, 1, 0, 0, 0, 0, 0, 0]
nodes = 100
dt = 0.02


def create_robot_model():
    eagle_yaml_dir = '/home/pepms/robotics/libraries/eagle_mpc_lib/yaml'

    robot = RobotWrapper.BuildFromURDF(join(eagle_yaml_dir, 'iris/description/iris.urdf'),
                                       package_dirs=[join(eagle_yaml_dir, '../..')],
                                       root_joint=pin.JointModelFreeFlyer())

    return robot


# Checking purposes
def integrate_trajectory(model, us, x0):
    data = model.createData()
    xs = [x0]

    tau_f = np.array([[0., 0., 0., 0.], [0., 0., 0., 0.], [1., 1., 1., 1.], [-0.22, 0.2, 0.22, -0.2],
                      [-0.13, 0.13, -0.13, 0.13], [-0.016, -0.016, 0.016, 0.016]])

    for idx, u in enumerate(us):
        q = xs[idx][:model.nq]
        v = xs[idx][model.nq:]

        tau = tau_f @ u

        a = pin.aba(model, data, q, v, tau)

        dq = v * dt + a * dt**2
        dv = a * dt

        q_next = pin.integrate(model, q, dq)
        v_next = v + dv

        x_next = np.concatenate([q_next, v_next])

        xs.append(x_next)

    return xs


def actuation_model():
    tau_f = np.array([[0., 0., 0., 0.], [0., 0., 0., 0.], [1., 1., 1., 1.], [-0.22, 0.2, 0.22, -0.2],
                      [-0.13, 0.13, -0.13, 0.13], [-0.016, -0.016, 0.016, 0.016]])

    u = casadi.SX.sym("u", 4)  # rotor velocities
    tau = tau_f @ u

    return casadi.Function('act_model', [u], [tau], ['u'], ['tau'])


def state_integrate(model):
    q = casadi.SX.sym("dq", model.nq)
    dq = casadi.SX.sym("q", model.nv)
    v = casadi.SX.sym("v", model.nv)
    dv = casadi.SX.sym("dv", model.nv)

    q_next = cpin.integrate(model, q, dq)
    v_next = v + dv

    dx = casadi.vertcat(dq, dv)
    x = casadi.vertcat(q, v)
    x_next = casadi.vertcat(q_next, v_next)

    return casadi.Function('integrate', [x, dx], [x_next], ['x', 'dx'], ['x_next'])


def state_difference(model):
    q0 = casadi.SX.sym("q0", model.nq)
    q1 = casadi.SX.sym("q1", model.nq)
    v0 = casadi.SX.sym("v0", model.nv)
    v1 = casadi.SX.sym("v1", model.nv)

    q_diff = cpin.difference(model, q0, q1)
    v_diff = v1 - v0

    x0 = casadi.vertcat(q0, v0)
    x1 = casadi.vertcat(q1, v1)
    x_diff = casadi.vertcat(q_diff, v_diff)

    return casadi.Function('difference', [x0, x1], [x_diff], ['x0', 'x1'], ['x_diff'])


def euler_integration(model, data, dt):
    nu = 4
    u = casadi.SX.sym("u", nu)

    # tau = casadi.vertcat(np.zeros(model.nv - nu), u)
    tau = actuation_model()(u)

    q = casadi.SX.sym("q", model.nq)
    v = casadi.SX.sym("v", model.nv)

    a = cpin.aba(model, data, q, v, tau)

    dq = v * dt + a * dt**2
    dv = a * dt

    x = casadi.vertcat(q, v)
    dx = casadi.vertcat(dq, dv)
    x_next = state_integrate(model)(x, dx)

    return casadi.Function('int_dyn', [x, u], [x_next], ['x', 'u'], ['x_next'])


def cost_quadratic_state_error(model):
    x_nom = [0, 0, 0, 0, 0, 0, 1, 0, 0, 0, 0, 0, 0]

    dx = casadi.SX.sym("dx", model.nv * 2)

    x_N = state_integrate(model)(x_nom, dx)
    e_goal = state_difference(model)(x_N, x_goal)

    cost = 0.5 * e_goal.T @ e_goal

    return casadi.Function('quad_cost', [dx], [cost], ['dx'], ['cost'])


def main():
    robot = create_robot_model()
    model = robot.model

    cmodel = cpin.Model(model)
    cdata = cmodel.createData()

    nv = cmodel.nv
    nu = 4

    x_nom = [0, 0, 0, 0, 0, 0, 1, 0, 0, 0, 0, 0, 0]

    # --------------OPTIMIZATION PROBLEM-----------

    opti = casadi.Opti()

    dxs = opti.variable(2 * nv, nodes + 1)  # state trajectory
    us = opti.variable(nu, nodes)  # control trajectory

    # Objective function
    obj = 0

    # State & Control regularization
    for i in range(nodes):
        x_i = state_integrate(cmodel)(x_nom, dxs[:, i])
        e_reg = state_difference(cmodel)(x_nom, x_i)
        obj += 1e-5 * 0.5 * e_reg.T @ e_reg + 1e-5 * 0.5 * us[:, i].T @ us[:, i]

    # obj += 1000 * cost_quadratic_state_error(cmodel)(dxs[:, nodes])

    opti.minimize(obj)

    # Dynamical constraints
    for i in range(nodes):
        x_i = state_integrate(cmodel)(x_nom, dxs[:, i])
        x_i_1 = state_integrate(cmodel)(x_nom, dxs[:, i + 1])
        f_x_u = euler_integration(cmodel, cdata, dt)(x_i, us[:, i])
        gap = state_difference(cmodel)(f_x_u, x_i_1)

        opti.subject_to(gap == [0] * 12)

    # Control constraints
    opti.subject_to(opti.bounded(0, us, 5))

    # Final constraint
    x_N = state_integrate(cmodel)(x_nom, dxs[:, nodes])
    e_goal = state_difference(cmodel)(x_N, x_goal)
    opti.subject_to(e_goal == [0] * 12)

    # Initial state
    x_0 = state_integrate(cmodel)(x_nom, dxs[:, 0])
    opti.subject_to(state_difference(cmodel)([0, 0, 0, 0, 0, 0, 1, 0, 0, 0, 0, 0, 0], x_0) == [0] * 12)

    # Warm start
    opti.set_initial(dxs, np.vstack([np.zeros(12) for _ in range(nodes + 1)]).T)
    opti.set_initial(us, np.vstack([np.zeros(4) for _ in range(nodes)]).T)

    opts = {'verbose': False}
    # opts['ipopt'] = {'max_iter': 1000, 'linear_solver': 'mumps', 'hessian_approximation': 'limited-memory', 'tol': 3.82e-6, 'mu_strategy': "adaptive"}
    opts['ipopt'] = {'max_iter': 1000, 'linear_solver': 'mumps', 'tol': 5e-6, 'mu_strategy': 'adaptive'}

    # Solver initialization
    opti.solver("ipopt", opts)  # set numerical backend

    try:
        sol = opti.solve()
    except:
        sol = opti.debug
        print()

    # --------------CURATE SOLUTION-----------
    xs_sol = []
    us_sol = []

    for idx, (dx_sol, u_sol) in enumerate(zip(sol.value(dxs).T, sol.value(us).T)):
        q = pin.integrate(model, np.array(x_nom)[:model.nq], dx_sol[:model.nv])
        v = dx_sol[model.nv:]
        xs_sol.append(np.concatenate([q, v]))
        us_sol.append(u_sol)

    q = pin.integrate(model, np.array(x_nom)[:model.nq], sol.value(dxs).T[nodes, :model.nv])
    v = sol.value(dxs).T[nodes, model.nv:]

    xs_sol.append(np.concatenate([q, v]))
    xs_pin = integrate_trajectory(model, us_sol, xs_sol[0])

    # --------------PLOTS-----------
    import matplotlib.pyplot as plt
    fig0, axs0 = plt.subplots(nrows=3)

    gaps = []
    for x_sol, x_pin in zip(xs_sol, xs_pin):
        q_diff = pin.difference(model, x_pin[:model.nq], x_sol[:model.nq])
        v_diff = x_sol[model.nq:] - x_pin[model.nq:]
        x_diff = np.concatenate([q_diff, v_diff])

        gaps.append(np.linalg.norm(x_diff))

    xs_sol_a = np.vstack(xs_sol)
    xs_pin_a = np.vstack(xs_pin)

    for idx, ax in enumerate(axs0):
        ax.plot(xs_sol_a[:, idx])
        ax.plot(xs_pin_a[:, idx])

    fig1, axs1 = plt.subplots(nrows=4)
    us_sol_a = np.vstack(us_sol)

    for idx, ax in enumerate(axs1):
        ax.plot(us_sol_a[:, idx])

    plt.figure()
    plt.plot(gaps)

    plt.show()


if __name__ == '__main__':
    main()