import casadi as ca
import numpy as np
from mpc_model import Model
from mpc_simulation import  Simulator
from mpc_solver import MPC
from mpc_plot import Plot
import matplotlib.pyplot as plt

def furuta_ode(t, x, u, p):
    """
    Furuta Pendulum
    """
    # Parameter configuration
    alpha = p[0]
    beta =  p[1]
    delta = p[2]
    gamma = p[3]

    dx1_dt = x[1]
    dx2_dt = 1 / (alpha * beta - gamma ** 2 + (beta ** 2 + gamma ** 2) * ca.sin(x[2]) ** 2) * (
                beta * gamma * (ca.sin(x[2] ** 2 - 1)) * ca.sin(x[2]) * x[1] ** 2 - 2 * beta ** 2 * ca.cos(
            x[2]) * ca.sin(x[2]) * x[1] * x[3] + beta * gamma * ca.sin(x[2]) * x[3] ** 2 - gamma * delta * ca.cos(
            x[2]) * ca.sin(x[2]) + beta * u[0] - gamma * ca.cos(x[2]) * u[1])
    dx3_dt = x[3]
    dx4_dt = 1 / (alpha * beta - gamma ** 2 + (beta ** 2 + gamma ** 2) * ca.sin(x[2]) ** 2) * (
                beta * (alpha + beta * ca.sin(x[2]) ** 2) * ca.cos(x[2]) * ca.sin(x[2]) * x[
            1] ** 2 + 2 * beta * gamma * (1 - ca.sin(x[2]) ** 2) * ca.sin(x[2]) * x[1] * x[3] - gamma ** 2 * ca.cos(
            x[2]) * ca.sin(x[2]) * x[3] ** 2 + delta * (alpha + beta * ca.sin(x[2]) ** 2) * ca.sin(
            x[2]) - gamma * ca.cos(x[2]) * u[0] + (alpha + beta * ca.sin(x[2]) ** 2) * u[1])

    rhs = [dx1_dt,
           dx2_dt,
           dx3_dt,
           dx4_dt
           ]
    return ca.vertcat(*rhs)

def set_constraint(N_pred):
    # set constraint
    lbx = [0, 0, -ca.pi, 0]
    ubx = [0, 0, -ca.pi, 0]
    lbx += [-ca.pi, -ca.inf, -ca.pi, -ca.inf, -ca.inf, -ca.inf] * N_pred
    ubx += [ca.pi, ca.inf, 2 * ca.pi, ca.inf, ca.inf, ca.inf] * N_pred

    x0 = [0, 0, -ca.pi, 0]
    x0 += [0, 0, 0, 0, 0, 0] * N_pred

    lbg = [0, 0, 0, 0] * N_pred
    ubg = [0, 0, 0, 0] * N_pred

    p = [0, 0, 0, 0, 0, 0, 0]

    return lbx, ubx, lbg, ubg, p, x0




if __name__ == "__main__":
    Nt = 1
    Nx = 4
    Nu = 2
    Np = 0
    Nz = 0

    delta_t = 0.01
    N_pred = 50
    N_sim = 100

    t_SX = ca.SX.sym("t_SX", Nt)
    x_SX = ca.SX.sym("x_SX", Nx)
    u_SX = ca.SX.sym("u_SX", Nu)
    p_SX = ca.SX.sym("p_SX", Np)
    z_SX = ca.SX.sym("z_SX", Nz)

    alpha = 0.0033472
    beta =  0.0038852
    delta = 0.097625
    gamma = 0.0024879

    para = [alpha,beta,delta,gamma]

    furuta_ode(t_SX, x_SX, u_SX, para)
    lbx, ubx, lbg, ubg, p, x0 = set_constraint(N_pred)


    xr_SX = ca.SX.sym("xr_SX", Nx)
    ur_SX = ca.SX.sym("ur_SX", Nu)
    Q = np.diag([10, 1, 100, 1])
    R = np.diag([1, 1])
    Q_f = np.diag([0, 0, 0, 0])
    stage_cost = (x_SX - xr_SX).T @ Q @ (x_SX - xr_SX) + (u_SX - ur_SX).T @ R @ (u_SX - ur_SX)    #  Lagrange term
    terminal_cost = (x_SX - xr_SX).T @ Q_f @ (x_SX - xr_SX)    #  Mayer term
    stage_cost_func = ca.Function("stage_cost_func",[x_SX, xr_SX, u_SX, ur_SX], [stage_cost])
    terminal_cost_func = ca.Function("terminal_cost_func",[x_SX, xr_SX], [terminal_cost])

    model = Model(t_SX, x_SX, u_SX, z_SX, p_SX, delta_t, para=para, ode=furuta_ode, alg=None, opt=None,
                  stage_cost_func=stage_cost_func, terminal_cost_func=terminal_cost_func)

    mpc_solver = MPC(model, N_pred)
    mpc_simulator = Simulator(model, mpc_solver, N_sim, lbx, ubx, lbg, ubg, p, x0)

    mpc_plot = Plot(mpc_simulator)
    mpc_plot.plot_single()