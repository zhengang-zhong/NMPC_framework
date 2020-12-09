import matplotlib.pyplot as plt
import casadi as ca
class Plot:
    def __init__(self, mpc_simulator):
        self.x_opt = mpc_simulator.x_trajectory
        self.u_opt = mpc_simulator.u_trajectory
        self.Nx = mpc_simulator.Nx
        self.Nu = mpc_simulator.Nu
        self.delta_t = mpc_simulator.delta_t
        self.N_sim = mpc_simulator.N_sim

    def plot_single(self):
        # Plot the solution
        Nx = self.Nx
        Nu = self.Nu
        delta_t = self.delta_t
        N_sim = self.N_sim

        tgrid = [delta_t * k for k in range(N_sim + 1)]

        plt.figure(1)
        plt.clf()

        legend = []
        for i in range (Nx):
            x_opt = self.x_opt[i::Nx]
            plt.plot(tgrid, x_opt)
            legend += ['x'+ str(i)]

        for i in range (Nu):
            u_opt = self.u_opt[i::Nu]
            plt.step(tgrid, ca.vertcat(ca.DM.nan(1), u_opt))
            legend += ['u'+ str(i)]
        plt.xlabel('t')

        plt.legend(legend)
        plt.grid()
        plt.show()


    def plot_multi(self):
        # Plot the solution
        Nx = self.Nx
        Nu = self.Nu
        delta_t = self.delta_t
        N_sim = self.N_sim

        tgrid = [delta_t * k for k in range(N_sim + 1)]

        plt.figure(1)
        plt.clf()
        for i in range (Nx):
            plt.subplot( str(Nx + Nu) + str(1) + str(i + 1) )
            x_opt = self.x_opt[i::Nx]
            plt.plot(tgrid, x_opt, 'r')
            plt.ylabel('x' + str(i + 1))

        for i in range(Nu):
            plt.subplot( str(Nx + Nu) + str(1) + str(Nx + i + 1) )
            u_opt = self.u_opt[i::Nu]
            plt.step(tgrid, ca.vertcat(ca.DM.nan(1), u_opt), 'r')
            plt.ylabel('u' + str(i + 1))
        plt.xlabel('t')
        plt.grid()
        plt.show()