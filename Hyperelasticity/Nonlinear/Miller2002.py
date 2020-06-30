"""
[Miller2002]
"""

import numpy as np
from sympy import *

from Hyperelasticity.HyperelasticModel import HyperelasticModel
from Hyperelasticity.utils import Common
from Hyperelasticity.utils import Evaluate
from Hyperelasticity.utils import MechanicalTests



class Miller2002(HyperelasticModel):

    c0 = symbols('c0')
    alpha = symbols('alpha')

    model = 'Nonlinear'
    environment = 'Ex vivo'
    tissue = 'Human'


    def __init__(self, name, _c0, _alpha):
        super().__init__(name)
        self._c0 = _c0
        self._alpha = _alpha


    def StrainEnergyDensityFunction(self):
        print(self.name)

        self.W = self.c0 * (self.l1**self.alpha + self.l2**self.alpha + self.l3**self.alpha - 3) / (self.alpha**2)
        # self.W = self.c0 * (self.l1 ** (2 * self.alpha) - 1) / (2 * self.alpha)
        print('W = ')
        print(self.W)
        # print(expand(W))
        # print(factor(W))

    def DefineFunctionsInStretches(self, _testFunc):
        # Mechanical test
        l, l_l1, l_l2, l_l3 = _testFunc()  # e.g. UniaxialTest

        self.Substitute(self.c0, self._c0)
        self.Substitute(self.alpha, self._alpha)
        self.Substitute(self.l1, l_l1)
        self.Substitute(self.l2, l_l2)
        self.Substitute(self.l3, l_l3)

        return l