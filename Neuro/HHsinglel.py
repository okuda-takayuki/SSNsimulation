import numpy as np

class HodgkinHuxleyModel:
    def __init__(self, dt=1e-3, solver="RK4"):
        self.C_m = 1.0
        self.g_Na = 120.0
        self.g_K = 36.0
        self.g_L = 0.3
        self.E_Na = 50.0
        self.E_K = -70.0
        self.E_L = -54.387

        self.solver = solver
        self.dt = dt

        self.states = np.array([-65, 0.05, 0.6, 0.32])
        self.I_m = None

    def Solvers(self, func, x, dt):
        if self.solver == "RK4":
            k1 = dt*func(x)
            k2 = dt*func(x+0.5*k1)
            k3 = dt*func(x + 0.5*k2)
            k4 = dt*func(x + k3)
            return x + (k1 + 2*k2 + 2*k3 + k4) / 6

        elif self.solver == "Euler":
            return x + dt*func(x)
        else:
            return None

def alpha_m(self, V):
    return 4.0*np.exp()

