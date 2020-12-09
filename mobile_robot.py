import casadi as ca
import numpy as np
from mpc_model import Model
from mpc_simulation import  Simulator
from mpc_solver import MPC
from mpc_plot import Plot
import matplotlib.pyplot as plt

def mobile_robot_ode(t, x, u, p = None):
    """
    Mobile robot
    """
    # Parameter configuratio

    dx_dt = u[0] * ca.cos(x[2])
    dy_dt = u[0] * ca.sin(x[2])
    dtheta_dt = u[1]
    rhs = [dx_dt,
           dy_dt,
           dtheta_dt
           ]
    return ca.vertcat(*rhs)

def set_constraint(Nx, Nu, N_pred):
    # set constraint
    lbx = [0, 0, 0]
    ubx = [0, 0, 0]
    lbx += [-ca.inf] * (Nx + Nu) * N_pred
    ubx += [ca.inf] * (Nx + Nu) * N_pred

    x0 = [0, 0, 0]
    x0 += [0] * (Nx + Nu) * N_pred

    lbg = [0] * Nx * N_pred
    ubg = [0] * Nx * N_pred

    p = [0, 10, 10, 0, 0, 0] # t_var, xr_var, ur_var

    return lbx, ubx, lbg, ubg, p, x0

if __name__ == "__main__":
    Nt = 1
    Nx = 3
    Nu = 2
    Np = 0
    Nz = 0

    delta_t = 0.01
    N_pred = 50
    N_sim = 300

    t_SX = ca.SX.sym("t_SX", Nt)
    x_SX = ca.SX.sym("x_SX", Nx)
    u_SX = ca.SX.sym("u_SX", Nu)
    p_SX = ca.SX.sym("p_SX", Np)
    z_SX = ca.SX.sym("z_SX", Nz)

    para = []

    mobile_robot_ode(t_SX, x_SX, u_SX, para)
    lbx, ubx, lbg, ubg, p, x0 = set_constraint(Nx, Nu, N_pred)

    xr_SX = ca.SX.sym("xr_SX", Nx)
    ur_SX = ca.SX.sym("ur_SX", Nu)
    Q = np.diag([1, 1, 1])
    R = np.diag([1, 1])
    Q_f = np.diag([100, 100, 100])
    stage_cost = (x_SX - xr_SX).T @ Q @ (x_SX - xr_SX) + (u_SX - ur_SX).T @ R @ (u_SX - ur_SX)    #  Lagrange term
    terminal_cost = (x_SX - xr_SX).T @ Q_f @ (x_SX - xr_SX)    #  Mayer term
    stage_cost_func = ca.Function("stage_cost_func",[x_SX, xr_SX, u_SX, ur_SX], [stage_cost])
    terminal_cost_func = ca.Function("terminal_cost_func",[x_SX, xr_SX], [terminal_cost])

    model = Model(t_SX, x_SX, u_SX, z_SX, p_SX, delta_t, para=para, ode=mobile_robot_ode, alg=None, opt=None,
                  stage_cost_func=stage_cost_func, terminal_cost_func=terminal_cost_func)

    mpc_solver = MPC(model, N_pred)
    mpc_simulator = Simulator(model, mpc_solver, N_sim, lbx, ubx, lbg, ubg, p, x0)

    mpc_plot = Plot(mpc_simulator)
    mpc_plot.plot_single()