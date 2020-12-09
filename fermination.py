import casadi as ca
import numpy as np
from mpc_model import Model
from mpc_simulation import  Simulator
from mpc_solver import MPC
from mpc_plot import Plot
import matplotlib.pyplot as plt

def fermination_ode(t, x, u, p):
    """
    Fermination model
    """
    k_dc = p[0]
    k_dm = p[1]
    s_i = p[2]

    T = u[0]

    x_lag = x[0]
    x_active = x[1]
    x_bottom = x[2]
    s = x[3]
    e = x[4]
    acet = x[5]
    diac = x[6]

    mu_x0 = ca.exp(108.31 - 31934.09 / (T+273.15))
    mu_eas = ca.exp(89.92 - 26589 / (T+273.15))
    mu_s0 = ca.exp( -41.92 + 11654.64/ (T+273.15))
    mu_lag = ca.exp(30.72 - 9501.54 / (T+273.15))
    k_dc = 0.000127672
    k_m = ca.exp( 130.16 - 38313 / (T+273.15))
    mu_D0 = ca.exp( 33.82 - 10033.28/ (T+273.15))
    mu_a0 = ca.exp(3.72 - 1267.24 / (T+273.15))
    k_s = ca.exp(-119.63 + 34203.95 / (T + 273.15))
    k_dm = 0.00113864

    mu_x = mu_x0 * s / (0.5 * s_i + e)
    mu_D = 0.5 * s_i * mu_D0 / (0.5 * s_i + e)
    mu_s = mu_s0 * s / (k_s + s)
    mu_a = mu_a0 * s / (k_s + s)
    f = 1 - e / (0.5 * s_i)

    dx_lag = - mu_lag * x_lag
    dx_active = mu_x * x_active - k_m * x_active + mu_lag * x_lag
    dx_bottom = k_m * x_active - mu_D * x_bottom
    ds = - mu_s * x_active
    de = mu_a * f * x_active
    dacet = - mu_eas * mu_s * x_active
    ddiac = k_dc * s * x_active - k_dm * diac * e
    rhs = [dx_lag,
           dx_active,
           dx_bottom,
           ds,
           de,
           dacet,
           ddiac
           ]
    return ca.vertcat(*rhs)

def set_constraint(Nx, Nu, N_pred):
    # set constraint
    lbx = [1.5, 0, 2, 122.41, 0, 10, 10]
    ubx = [1.5, 0, 2, 122.41, 0, 10, 10]
    lbx += [-ca.inf] * (Nx + Nu) * N_pred
    ubx += [ca.inf] * (Nx + Nu) * N_pred

    x0 = [1.5, 0, 2, 22.41, 0, 0, 0]
    x0 += [0] * (Nx + Nu) * N_pred

    lbg = [0] * Nx * N_pred
    ubg = [0] * Nx * N_pred

    p = [0, 0, 1, 0, 0, 60, 1.05, 0, 10] # t_var, xr_var, ur_var

    return lbx, ubx, lbg, ubg, p, x0

if __name__ == "__main__":
    Nt = 1
    Nx = 7
    Nu = 1
    Np = 0
    Nz = 0

    delta_t = 1
    N_pred = 50
    N_sim = 100

    t_SX = ca.SX.sym("t_SX", Nt)
    x_SX = ca.SX.sym("x_SX", Nx)
    u_SX = ca.SX.sym("u_SX", Nu)
    p_SX = ca.SX.sym("p_SX", Np)
    z_SX = ca.SX.sym("z_SX", Nz)

    k_dc = 0.000127672
    k_dm =  0.00113864
    s_i = 22.41

    para = [k_dc, k_dm, s_i]


    fermination_ode(t_SX, x_SX, u_SX, para)
    lbx, ubx, lbg, ubg, p, x0 = set_constraint(Nx, Nu, N_pred)


    xr_SX = ca.SX.sym("xr_SX", Nx)
    ur_SX = ca.SX.sym("ur_SX", Nu)

    stage_cost = 0.001**(-17) * ca.exp(2.31 * u_SX)   - 100 * (u_SX - ur_SX) #  Lagrange term
    terminal_cost = 10 * x_SX[4] - 1.16 * ca.exp(48 * x_SX[5] - 66.77) -5.73 * ca.exp(11 * x_SX[6] - 11.51)    #  Mayer term
    stage_cost_func = ca.Function("stage_cost_func",[x_SX, xr_SX, u_SX, ur_SX], [stage_cost])
    terminal_cost_func = ca.Function("terminal_cost_func",[x_SX, xr_SX], [terminal_cost])

    model = Model(t_SX, x_SX, u_SX, z_SX, p_SX, delta_t, para=para, ode=fermination_ode, alg=None, opt=None,
                  stage_cost_func=stage_cost_func, terminal_cost_func=terminal_cost_func)

    solver_opt = {}
    solver_opt['print_time'] = False
    solver_opt['ipopt'] = {
        'max_iter': 500,
        'print_level': 1,
        'acceptable_tol': 1e-6,
        'acceptable_obj_change_tol': 1e-6
    }

    mpc_solver = MPC(model, N_pred, solver_opt = solver_opt)
    mpc_simulator = Simulator(model, mpc_solver, N_sim, lbx, ubx, lbg, ubg, p, x0)
    mpc_plot = Plot(mpc_simulator)
    mpc_plot.plot_multi()