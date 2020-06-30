"""
[Laksari2012] from [Morin2017]
"""

import numpy as np
from sympy import *

from Hyperelasticity.HyperelasticModel import HyperelasticModel
from Hyperelasticity.utils import Common
from Hyperelasticity.utils import Evaluate
from Hyperelasticity.utils import MechanicalTests


class Laksari2012(HyperelasticModel):

    c10 = symbols('c10')
    c01 = symbols('c01')
    c11 = symbols('c11')
    K = symbols('K')
    alpha = symbols('alpha')
    v = symbols('v')

    model = 'Nonlinear'
    environment = 'Ex vivo'
    tissue = 'Bovine'


    def __init__(self, name, _c10, _c01, _c11, _K, _alpha, _v):
        super().__init__(name)
        self._c10 = _c10
        self._c01 = _c01
        self._c11 = _c11
        self._K = _K
        self._alpha = _alpha
        self._v = _v


    def StrainEnergyDensityFunction(self):
        # super().StrainEnergyDensityFunction(self.Ic, self.IIc, self.IIIc)

        print(self.name)

        # incompressibility (in the direction of compression)
        self.J = symbols('J')
        self.J = self.l1 ** (1.0 - 2.0 * self.v)
        self.Ic   = self.Ic * self.J ** (-2 / 3)
        self.IIc  = self.IIc * self.J ** (-2 / 3)
        self.IIIc = self.IIIc * self.J ** (-2 / 3)

        # isochoric component
        self.W = self.c10 * (self.Ic - 3) + self.c01 * (self.IIc - 3) + self.c11 * (self.Ic - 3) * (self.IIc - 3)

        # volumetric component
        self.W = self.W + self.K / pow(self.alpha, 2) * (self.alpha * log(self.J) + pow(self.J,-self.alpha) - 1)

        print('W = ')
        print(self.W)
        # print(expand(W))
        # print(factor(W))


    def DefineFunctionsInStretches(self, _testFunc):
        # Mechanical test
        l, l_l1, l_l2, l_l3 = _testFunc()  # e.g. UniaxialTest

        self.Substitute(self.c10, self._c10)
        self.Substitute(self.c01, self._c01)
        self.Substitute(self.c11, self._c11)
        self.Substitute(self.K, self._K)
        self.Substitute(self.alpha, self._alpha)
        self.Substitute(self.v, self._v)
        self.Substitute(self.l1, l_l1)
        self.Substitute(self.l2, l_l2)
        self.Substitute(self.l3, l_l3)

        return l

