import casadi as ca

class MPC:
    def __init__(self,model, N_pred, solver_opt = None):
        self.model = model
        self.N_pred = N_pred
        self.solver = self.multiple_shooting(model,N_pred, solver_opt)
        self.type = 'ms'

    # def single_shooting(self, model, N_pred, solver_opt=None, g_model=None):

    def multiple_shooting(self, model, N_pred, solver_opt=None):
        '''
        Solve the ODE for the NLP solver with multiple shooting optimization formulation.

        Args:
            model: An instance of model class.
            N_pred: Number of control intervals.
            solver_opt: Option dictionary for the solver.
            g_model: Extra equality and inequality constraints.

        Returns:
            solver: A casadi NLP solver for the given control problem.

        TODO:
        1. Stage cost for time-variant reference
        -> generate x_r = ca.SX.sym('x_r', Nx, N+1), same for u_r
        2. Extra equality and inequality constraints should be incorporated from g_model.
        '''
        Nx = model.Nx
        Nu = model.Nu
        Nt = model.Nt
        Np = model.Np

        delta_t = model.delta_t

        OPT_variables = []
        g = []
        p = []
        obj = 0

        t_var = ca.SX.sym('ti', Nt)
        ti_var = t_var
        xr_var = ca.SX.sym('xr', Nx)
        ur_var = ca.SX.sym('ur', Nu)
        p_var = ca.SX.sym('p', Np)

        xi_var = ca.SX.sym('x0', Nx)

        stage_cost_func = model.stage_cost_dis_func
        terminal_cost_func = model.terminal_cost_func
        fn = model.ode_dis_func

        OPT_variables += [xi_var]
        for i in range(N_pred):
            ui_var = ca.SX.sym('u_' + str(i), Nu)
            OPT_variables += [ui_var]

            # Integrate till the end of the interval
            qi = stage_cost_func(ti_var, xi_var, xr_var, ui_var, ur_var)
            xi_end_var = fn(ti_var, xi_var, ui_var, p_var)
            obj += qi

            # New NLP variable for state at end of interval
            ti_var += delta_t
            xi_var = ca.SX.sym('x_' + str(i + 1), Nx)
            OPT_variables += [xi_var]

            # Add equality constraint
            g += [xi_end_var - xi_var]
        obj += terminal_cost_func(xi_var, xr_var)

        p = [t_var, xr_var, ur_var]

        nlp_prob = {
            'f': obj,
            'x': ca.vertcat(*OPT_variables),
            'g': ca.vertcat(*g),
            'p': ca.vertcat(*p)
        }

        print(ca.vertcat(*OPT_variables).shape)

        if solver_opt is None:
            solver_opt = {}
            solver_opt['print_time'] = False
            solver_opt['ipopt'] = {
                'max_iter': 500,
                'print_level': 3,
                'acceptable_tol': 1e-6,
                'acceptable_obj_change_tol': 1e-6
            }

        solver = ca.nlpsol("solver", "ipopt", nlp_prob, solver_opt)

        return solver

        # def orthogonal_collocation(self, model, N_pred, solver_opt=None, g_model=None):