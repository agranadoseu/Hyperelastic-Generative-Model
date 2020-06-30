"""
Implementation class of a Neo-Hookean meta model
"""

import numpy as np
from scipy.optimize import least_squares

from Optimisation.HyperMetaModel import HyperMetaModel


class NeoHookeanMetaModel(HyperMetaModel):

    name = "Neo-Hookean meta model"
    wn = ["mu"]
    w = [0.]
    w0 = np.array([1000.0])
    #bounds = (0.001, 100000000)
    bounds = (0.001, np.inf)

    def __init__(self):
        super().__init__()

    def model_Psi(self, w, x):
        self.w = w
        return self.w[0]/2.0 * (x**2.0 + 2.0*x**(-1.0) - 3.0)

    def jacobian(self, w, x, y):
        self.w = w
        J = np.empty((x.size, self.w.size))
        core = self.w[0]/2.0 * (x**2.0 + 2.0*x**(-1.0) - 3.0)
        J[:, 0] = core / (self.w[0]*core + 0.001)
        return J

    def optimise(self, xm, ym_log):
        res = least_squares(self.fun_residual, self.w0, jac=self.jacobian, bounds=self.bounds, args=(xm, ym_log), verbose=0)
        self.w = res.x
        print(self.name, np.around(res.x, decimals=2), np.around(res.cost, decimals=2))

        ylog_test = self.model_log(res.x, self.x_test) - 0.001
        y_test = self.model_Psi(res.x, self.x_test)

        return res.x, res.cost, self.x_test, ylog_test, y_test



class NeoHookeanPenaltyMetaModel(HyperMetaModel):

    name = "Neo-Hookean Penalty meta model"
    wn = ["mu"]
    w = np.array([0.])
    w0 = np.array([10000000.0])
    bounds = (-100000000000000.0, 100000000000000.0)

    def __init__(self):
        super().__init__()

    def model_Psi(self, w, x):
        self.w = w
        return self.w[0]/2.0 * (x**2.0 + 2.0*x**(-1.0) - 3.0)

    def jacobian(self, w, x, y):
        self.w = w
        J = np.empty((x.size, self.w.size))
        core = self.w[0]/2.0 * (x**2.0 + 2.0*x**(-1.0) - 3.0)
        J[:, 0] = core / (self.w[0]*core + 0.001)
        return J

    def fun_residual(self, w, x, y):
        self.w = w
        # EXPONENTIAL DECAY (ED) = penalise extreme low values close to zero:  N(t) = N0 * e^{-lt}
        ED = np.zeros_like(x)
        if abs(self.w[0]) <= 1.0:
            ED = 1000. * np.exp(-2.*abs(self.w[0])) * np.ones_like(x)

        # HESSIAN POSITIVE (HP) = penalty term for negative Hessians
        core = self.model_Psi(self.w, x)
        valid = np.where(core != 0.)[0] # avoid zero
        H = np.zeros_like(x)
        H[valid] = self.w[0]*(1.+2.*x[valid]**(-3.))/core[valid] - (self.w[0]*(x[valid]-x[valid]**(-2)))**2/core[valid]**2
        zeros = np.zeros_like(x)
        HP = np.amax([zeros,H], axis=0)**2

        return self.model_log(self.w, x) - y + HP + ED

    def optimise(self, xm, ym_log):
        res = least_squares(self.fun_residual, self.w0, jac=self.jacobian, bounds=self.bounds, args=(xm, ym_log), loss='cauchy', f_scale=0.1, verbose=0)
        self.w = res.x
        print(np.around(res.x, decimals=2), np.around(res.cost,decimals=2))

        ylog_test = self.model_log(res.x, self.x_test) - 0.001
        y_test = self.model_Psi(res.x, self.x_test)

        return res.x, res.cost, self.x_test, ylog_test, y_test