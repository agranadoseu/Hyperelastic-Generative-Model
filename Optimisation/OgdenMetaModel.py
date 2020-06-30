"""
Implementation class of an Ogden meta model
"""

import numpy as np
from scipy.optimize import least_squares
from scipy.optimize import Bounds

from Optimisation.HyperMetaModel import HyperMetaModel

class OgdenMetaModel(HyperMetaModel):

    N = 1
    lb, ub = [], []

    name = "Ogden 1-term meta model"
    wn = []
    w = []
    # w0 = np.array([], dtype=np.float64)
    w0 = []
    w0b = []
    bounds = 0
    num_bounds = 0

    def __init__(self, N=1, aim='lower'):
        super().__init__()
        self.N = N
        self.aim = aim

        # init parameters
        for n in range(self.N):
            self.wn.append('m' + str(n + 1))    # mu
            self.wn.append('a' + str(n + 1))    # alpha
            self.w.append(0.)   # mu
            self.w.append(0.)   # alpha
            # self.w0 = np.append(self.w0, [1000.], axis=0)   # mu
            # self.w0 = np.append(self.w0, [10.], axis=0)      # alpha

            # bounds
            # self.lb.append(0.1)       # mu
            # self.lb.append(0.1)       # alpha
            # self.ub.append(np.inf)   # mu
            # self.ub.append(np.inf)   # alpha

        # self.bounds = Bounds(self.lb, self.ub)
        # self.bounds = [tuple(self.lb), tuple(self.ub)]

        # set bounds depending on the number of terms
        inf = np.inf
        if self.N == 1:
            if self.aim == 'lower':
                self.name += ' [for low terms]'
                self.num_bounds = 2
                self.lb, self.ub, self.w0b = [], [], []
                # self.lb.append([0.1, 0.1]),     self.ub.append([inf,inf]),      self.w0b.append([1000.,10.])
                # self.lb.append([-inf,-inf]),    self.ub.append([-0.1,-2.1]),    self.w0b.append([-1000.,-10.])    # good Ogden
                self.lb.append([0.1, 0.1]),       self.ub.append([inf, 5.0]),    self.w0b.append([1000., .1])
                self.lb.append([-inf, -5.0]),    self.ub.append([-0.1, -2.1]),   self.w0b.append([-1000., -2.1])  # good non-Ogden

            if self.aim == 'higher':
                self.name += ' [for high terms]'
                self.num_bounds = 8
                self.lb, self.ub, self.w0b = [], [], []
                self.lb.append([0.1, 0.1]), self.ub.append([inf, 5.0]), self.w0b.append([1000., .1])
                self.lb.append([0.1, 0.1]), self.ub.append([inf, 10.0]), self.w0b.append([1000., .1])
                self.lb.append([0.1, 0.1]), self.ub.append([inf, 20.0]), self.w0b.append([1000., .1])
                self.lb.append([0.1, 0.1]), self.ub.append([inf, 50.0]), self.w0b.append([1000., .1])
                self.lb.append([-inf, -5.0]), self.ub.append([-0.1, -2.1]), self.w0b.append([-1000., -2.1])
                self.lb.append([-inf, -10.0]), self.ub.append([-0.1, -2.1]), self.w0b.append([-1000., -2.1])
                self.lb.append([-inf, -20.0]), self.ub.append([-0.1, -2.1]), self.w0b.append([-1000., -2.1])
                self.lb.append([-inf, -50.0]), self.ub.append([-0.1, -2.1]), self.w0b.append([-1000., -2.1])
            self.bounds = 0
        elif self.N == 2:
            self.num_bounds = 2
            self.lb.append([0.1,0.1,0.1,0.1]),      self.ub.append([inf,inf,inf,inf]),      self.w0b.append([1000.,10.,1000.,10.])
            self.lb.append([-inf,-inf,-inf,-inf]),  self.ub.append([-0.1,-0.1,-0.1,-0.1]),  self.w0b.append([-1000.,-10.,-1000.,-10.])
            self.bounds = 0
        elif self.N == 3:
            self.num_bounds = 2
            self.lb.append([0.1,0.1,0.1,0.1,0.1,0.1]),      self.ub.append([inf,inf,inf,inf,inf,inf]),      self.w0b.append([1000.,10.,1000.,10.,1000.,10.])
            self.lb.append([-inf,-inf,-inf,-inf,-inf,-inf]),self.ub.append([-0.1,-0.1,-0.1,-0.1,-0.1,-0.1]),self.w0b.append([-1000.,-10.,-1000.,-10.,-1000.,-10.])
            self.bounds = 0


    def model_Psi(self, w, x):
        '''
                    N
            Psi = sum   m_p/a_p [ x^a_p + 2*x^(-a_p/2) - 3 ]
                    n=1
        '''
        self.w = w
        Psi = 0.
        for n in range(self.N):
            Psi += (self.w[n] / self.w[n+1]) * (x**self.w[n+1] + 2.*x**(-self.w[n+1]/2.) - 3.)
        return Psi

    def jacobian(self, w, x, y):
        '''
                                1/a_p [ x^a_p + 2*x^(-a_p/2) - 3 ]
            J[:n] =  -------------------------------------------------------
                          N
                        sum   m_p/a_p [ x^a_p + 2*x^(-a_p/2) - 3 ] + 0.001
                          n=1

                         m_p/a_p * (x^a_p*ln(x) - x^{-0.5*a_p}*ln(x)) - m_p/a_p^{2} * (x^a_p + 2*x^{-0.5*a_p} - 3 )
            J[:n+1] =  ---------------------------------------------------------------------------
                             N
                            sum   m_p/a_p [ x^a_p + 2*x^(-a_p/2) - 3 ] + 0.001
                            n=1
        '''
        self.w = w
        J = np.empty((x.size, self.w.size))

        core = 0.
        for n in range(self.N):
            core += (self.w[n] / self.w[n+1]) * (x**self.w[n+1] + 2.*x**(-self.w[n+1]/2.) - 3.)

        for n in range(self.N):
            J[:, n] = ((1./self.w[n+1]) * (x**self.w[n+1] + 2.*x**(-self.w[n+1]/2.) - 3.)) / (core + 0.001)
            J[:, n+1] = (  ( (self.w[n]/self.w[n+1]) * (x**self.w[n+1]*np.log(x) - x**(-0.5*self.w[n+1])*np.log(x)) )
                         - ( (self.w[n]/self.w[n+1]**2.) * (x**self.w[n+1] + 2.*x**(-0.5*self.w[n+1]) - 3.) )) \
                        / (core + 0.001)
        return J

    def optimise(self, xm, ym_log):
        # iterate through valid parameter space and choose best
        best_w = []
        best_c = np.inf
        str_bounds = ''
        for s in range(self.num_bounds):
            self.bounds = [tuple(self.lb[s]), tuple(self.ub[s])]
            self.w0 = np.array(self.w0b[s], dtype=np.float64)
            res = least_squares(self.fun_residual, self.w0, jac=self.jacobian, bounds=self.bounds, args=(xm, ym_log), verbose=0)
            str_bounds += 's={} w={} c={}     '.format(self.bounds, self.w0, np.around(res.cost, decimals=2))
            if res.cost < best_c:
                best_w = res.x
                best_c = res.cost
        # print(str_bounds, best_w, best_c)
        print(self.name, str_bounds, np.around(best_w, decimals=2), np.around(best_c, decimals=2))

        ylog_test = self.model_log(best_w, self.x_test) - 0.001
        y_test = self.model_Psi(best_w, self.x_test)

        return best_w, best_c, self.x_test, ylog_test, y_test