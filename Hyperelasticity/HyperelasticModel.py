"""
Abstract class of all hyperelastic models
"""

import numpy as np
from sympy import *

from Hyperelasticity.utils import Common
from Hyperelasticity.utils import Symbolic
from Hyperelasticity.utils import Evaluate
from Hyperelasticity.utils import Postulates
from Hyperelasticity.utils import PlotMaterial

s = Common.sample
r = Common.lambdaRange

class HyperelasticModel:

    W = symbols('W')

    stretches = np.linspace(r[0], r[1], s)

    def __init__(self, name):
        self.name = name


    # def StrainEnergyDensityFunction(self, Ic, IIc, IIIc):
    #     self.Ic = Ic
    #     self.IIc = IIc
    #     self.IIIc = IIIc


    def Substitute(self, _sym, _val):
        # W
        self._W = self._W.subs(_sym, _val)

        # dW
        for i in range(3):
            self._dW[i] = self._dW[i].subs(_sym, _val)

        # H
        for i in range(3):
            for j in range(3):
                self._H[i, j] = self._H[i, j].subs(_sym, _val)

        # sigma
        for i in range(3):
            self._sigma[i] = self._sigma[i].subs(_sym, _val)


    def Compute(self):
        self.l1, self.l2, self.l3 = Common.sym_Stretches()
        self.Ic, self.IIc, self.IIIc = Common.sym_Invariants(self.l1, self.l2, self.l3)

        # model
        self.StrainEnergyDensityFunction()

        # energy gradient
        self.dW = Symbolic.ComputeEnergyGradient(self.W, self.l1, self.l2, self.l3)

        # energy Hessian
        self.H = Symbolic.ComputeEnergyHessian(self.dW, self.l1, self.l2, self.l3)

        # Cauchy stress (sigma)
        self.sigma = Symbolic.ComputeCauchyStress(self.dW, self.l1, self.l2, self.l3)

        # substitutions
        self._W = self.W
        self._dW = self.dW
        self._H = self.H
        self._sigma = self.sigma


    def Evaluate(self, mechTest, stretchBound):
        # Evaluate
        # print('W = ', self._W)
        # print('dW = ', self._dW)
        l = self.DefineFunctionsInStretches(mechTest)
        # print('W(uniaxial) = ', self._W)
        # print('dW(uniaxial) = ', self._dW)

        self.stretches = np.linspace(stretchBound[0], stretchBound[1], s)

        self.W_v, self.dW_v, self.H_v, self.sigma_v = Evaluate.Functions(l, self.stretches, self._W, self._dW, self._H, self._sigma)
        value = Evaluate.SinglePoint(self._W, l, 5.0)

        P, dWoffset = [], 0.
        P, dWoffset = Postulates.verify(l, self.stretches, self._W, self._dW, self._H, self._sigma)
        print(P)
        print(dWoffset)

        return self.stretches, self.W_v, self.dW_v, self.H_v, self.sigma_v, P, dWoffset


    def Plot(self):
        # Plot
        PlotMaterial.Functions(self.name, self.stretches, self.W_v, self.dW_v, self.H_v, self.sigma_v)
