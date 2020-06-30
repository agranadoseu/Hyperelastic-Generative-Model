"""
Abstract class for hyperelastic metamodels
Specific implementations include:
    - Neo-Hookean
    - Mooney-Rivlin
    - Ogden 1-, 2- and 3-terms
"""

import numpy as np
from scipy.optimize import least_squares

import matplotlib.pyplot as plt


class HyperMetaModel:

    name = "HyperMetaModel abstract class"
    wn = [] # param names
    w = []  # unknown param values
    w0 = [] # initial param values
    bounds = () # bounds of parameters for optimisation

    S = 501  # sample size
    # x_test = np.linspace(0.001, 5, S)
    x_test = np.linspace(0.000001, 5, S)

    def __init__(self):
        return

    def model_Psi(self, w, x):
        """
        Hyperelastic function of meta model
        :param x: values of independent variables
        :return: strain energy density functions
        """
        return 1.

    def model_log(self, w, x):
        """
        Hyperelastic function of meta model in log space
        :param x: values of independent variables
        :return: strain energy density functions in log space
        """
        return np.log( self.model_Psi(w, x) + 0.001)

    def model_loginv(self, Wlog):
        """
        Inverse transform from log space back to strain energy density function space
        :param logW: stran energy density function in log space
        :return:
        """
        What = np.exp(Wlog) - 0.001
        return What

    def fun_residual(self, w, x, y):
        """
        Function of residuals
        :param x: values of independent variables
        :param y: measurement values
        :return:
        """
        return self.model_log(w, x) - y

    def jacobian(self, w, x, y):
        """
        Compute Jacobian in a closed form
        i.e. partial derivatives of model wrt to w parameters
        :param x: values of independent variables
        :return:
        """
        return 1.

    def optimise(self, xm, ym_log):
        """
        Optimisation method implementation
        :param xm: principal stretches data points to fit model to
        :param ym_log: strain energy density function in log space data points to fit model to
        :return:
        """
        return []

    def plot(self, xm, mean, variance, ym_log, xt, yt_log, yt):
        fig = plt.figure(1)
        title = self.name + "\n"
        for i in range(len(self.w)):
            title += self.wn[i] + " " + str(self.w[i])
        fig.suptitle(title, fontsize=14)

        # log space
        axis = plt.subplot(121)
        plt.plot(xm, mean, '-', markersize=4, alpha=1.0, color='black', )
        plt.vlines(xm, mean - 2. * np.sqrt(variance), mean + 2. * np.sqrt(variance), color='black', lw=2, alpha=0.05)
        plt.plot(xm, ym_log, 'o', markersize=6, label='predicted sample', alpha=0.3)
        plt.plot(xt, yt_log, label=self.name)
        plt.xlabel("x")
        plt.ylabel("y")
        # plt.ylim(0, 4e7)
        plt.legend(loc='lower right')

        axis = plt.subplot(122)
        plt.plot(xm, self.model_loginv(mean), '-', markersize=4, alpha=1.0, color='black', )
        plt.vlines(xm, self.model_loginv(mean - 2. * np.sqrt(variance)),
                   self.model_loginv(mean + 2. * np.sqrt(variance)), color='black', lw=2, alpha=0.05)
        plt.plot(xm, self.model_loginv(ym_log), 'o', markersize=6, label='predicted sample', alpha=0.3)
        plt.plot(xt, yt, label=self.name)
        plt.xlabel("x")
        plt.ylabel("y")
        # axis.set_ylim([0, ym[0]])
        # axis.set_ylim([0, 70000])
        axis.set_xlim([min(xm), max(xm)])
        axis.set_ylim([0, max(self.model_loginv(ym_log))])
        plt.legend(loc='lower right')

        mng = plt.get_current_fig_manager()
        mng.window.state('zoomed')
        plt.show()
        