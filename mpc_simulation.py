

class Simulator:
    def __init__(self, model, mpc_solver, N_sim, lbx, ubx, lbg, ubg, p, x0 = None, para = None):
        Nx = model.Nx
        Nu = model.Nu
        delta_t = model.delta_t
        N_pred = mpc_solver.N_pred
        solver = mpc_solver.solver


        self.Nx = Nx
        self.Nu = Nu
        self.delta_t = delta_t
        self.N_sim = N_sim
        self.N_pred = N_pred


        if para == None:
            para = model.para    # Nominal parameter.

        system_integrator = model.system_integrator

        x_trajectory = []
        u_trajectory = []
        x_opt_past = []

        # Define initial guess, constraints of decision variables and other constraints
        nl = {}
        nl['lbx'] = lbx
        nl['ubx'] = ubx
        nl['lbg'] = lbg
        nl['ubg'] = ubg
        nl['p'] = p

        t_init = p[0]
        # Multiple shooting
        if mpc_solver.type == 'ms':
            if x0 is None:
                nl['x0'] = [0] * Nx
                nl['x0'] += [0] * (Nx + Nu) * N_pred
            else:
                nl['x0'] = x0
            x_trajectory += lbx[0:Nx]  # For multiple shooting, first Nx elements belong to the initial state

        # Start simulation
        for i in range(N_sim):
            sol = solver(**nl)
            x_opt_past = sol['x'].full().flatten()

            # Find optimal input
            opt_u = x_opt_past[Nx:Nx+Nu].tolist()
            u_trajectory += opt_u

            # Apply the optimal input to the real system
            para_real = [t_init + delta_t * i] + opt_u + para
            x_next_real = system_integrator(x0=x_trajectory[-Nx:], p=para_real)['xf']    # TODO: If it is a DAE, function should contain algebraic z.
            x_trajectory += x_next_real.full().flatten().tolist()    #  ca.DM -> list

            # Update solver
            nl['x0'] = x_opt_past
            nl['lbx'][:Nx] = x_next_real.full().flatten().tolist()
            nl['ubx'][:Nx] = x_next_real.full().flatten().tolist()
            nl['p'][0] += (i+1) * delta_t


        self.x_trajectory = x_trajectory
        self.u_trajectory = u_trajectory