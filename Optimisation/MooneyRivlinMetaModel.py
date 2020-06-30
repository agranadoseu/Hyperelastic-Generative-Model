"""
Implementation class of a Mooney-Rivlin meta model
"""

import numpy as np
from scipy.optimize import least_squares
from scipy.optimize import Bounds

from Optimisation.HyperMetaModel import HyperMetaModel

class MooneyRivlinMetaModel(HyperMetaModel):

    name = "Mooney-Rivlin meta model"
    wn = ["C1", "C2"]
    w = [0., 0.]
    w0 = np.array([1000.0, 1000.0])
    # w0 = np.array([1000.0, 500.0])
    # bounds = Bounds([0.001, 0.001], [100000000, 100000000])
    # bounds = [(0.001, 0.001), (100000000, 100000000)]
    bounds = [(0.1, 0.1), (np.inf, np.inf)]
    # bounds = [(500, 10), (np.inf, 800)]

    def __init__(self):
        super().__init__()

    def model_Psi(self, w, x):
        self.w = w
        return self.w[0] * (x**2.0 + 2.0*x**(-1.0) - 3.0) + self.w[1] * (2.0*x + x**(-2.0) - 3.0)

    def jacobian(self, w, x, y):
        self.w = w
        J = np.empty((x.size, self.w.size))
        core = self.w[0] * (x**2.0 + 2.0*x**(-1.0) - 3.0) + self.w[1] * (2.0*x + x**(-2.0) - 3.0)
        J[:, 0] = (x**2.0 + 2.0*x**(-1.0) - 3.0) / (core + 0.001)
        J[:, 1] = (2.0*x + x**(-2.0) - 3.0) / (core + 0.001)
        return J

    def optimise(self, xm, ym_log):
        res = least_squares(self.fun_residual, self.w0, jac=self.jacobian, bounds=self.bounds, args=(xm, ym_log), verbose=0)
        self.w = res.x
        print(self.name, np.around(res.x, decimals=2), np.around(res.cost, decimals=2))

        ylog_test = self.model_log(res.x, self.x_test) - 0.001
        y_test = self.model_Psi(res.x, self.x_test)

        return res.x, res.cost, self.x_test, ylog_test, y_test




class MooneyRivlinPenaltyMetaModel(HyperMetaModel):

    name = "Mooney-Rivlin Penalty meta model"
    wn = ["C1", "C2"]
    w = np.array([0., 0.])
    w0 = np.array([10000000.0, 10000000.0])
    # bounds = Bounds([0.001, 0.001], [100000000, 100000000])
    # bounds = [(0.001, 0.001), (100000000, 100000000)]
    bounds = [(-100000000000000.0, -100000000000000.0), (np.inf, np.inf)]
    # bounds = [(0.001, 0.001), (np.inf, np.inf)]

    def __init__(self):
        super().__init__()

    def model_Psi(self, w, x):
        self.w = w
        return self.w[0] * (x**2.0 + 2.0*x**(-1.0) - 3.0) + self.w[1] * (2.0*x + x**(-2.0) - 3.0)

    def jacobian(self, w, x, y):
        self.w = w
        J = np.empty((x.size, self.w.size))
        core = self.w[0] * (x**2.0 + 2.0*x**(-1.0) - 3.0) + self.w[1] * (2.0*x + x**(-2.0) - 3.0)
        J[:, 0] = (x**2.0 + 2.0*x**(-1.0) - 3.0) / (core + 0.001)
        J[:, 1] = (2.0*x + x**(-2.0) - 3.0) / (core + 0.001)
        return J

    def fun_residual(self, w, x, y):
        self.w = w
        # if self.w[0] == 0.0 or self.w[1] == 0.0:
        #     return np.ones_like(x)*1000.0
        # EXPONENTIAL DECAY (ED) = penalise extreme low values close to zero:  N(t) = N0 * e^{-lt}
        ED = np.zeros_like(x)
        if abs(self.w[0]) <= 1.0:
            ED += 1. * np.exp(-2. * abs(self.w[0])) * np.ones_like(x)
        if abs(self.w[1]) <= 1.0:
            ED += 1. * np.exp(-2. * abs(self.w[1])) * np.ones_like(x)

        # HESSIAN POSITIVE (HP) = penalty term for negative Hessians
        core = self.model_Psi(self.w, x)
        valid = np.where(core != 0.)[0] # avoid zero
        H = np.zeros_like(x)
        H[valid] = (2.*self.w[0]*(1.+2.*x[valid]**(-3.)) + 6.*self.w[1]*x[valid]**(-4.))/core[valid] - \
                   (2.*self.w[0]*(x[valid]-x[valid]**(-2)) + 2.*self.w[1]*(1.-x[valid]**(-3)))**2/core[valid]**2
        zeros = np.zeros_like(x)
        HP = np.amax([zeros,H], axis=0)**2

        # additional penalty (results still show Hessian as negative)
        # negative = np.where(H < 0)[0]
        # nHP = np.zeros_like(H)
        # nHP[negative] = 1.0 * H[negative]

        return self.model_log(self.w, x) - y + HP + ED

    def optimise(self, xm, ym_log):
        res = least_squares(self.fun_residual, self.w0, jac=self.jacobian, bounds=self.bounds, args=(xm, ym_log), loss='cauchy', f_scale=0.1, verbose=0)
        self.w = res.x
        print(np.around(res.x, decimals=2), np.around(res.cost, decimals=2))

        ylog_test = self.model_log(res.x, self.x_test) - 0.001
        y_test = self.model_Psi(res.x, self.x_test)

        return res.x, res.cost, self.x_test, ylog_test, y_test