"""
[Schiavone2009] from [Morin2017]
"""

import numpy as np
from sympy import *

from Hyperelasticity.HyperelasticModel import HyperelasticModel
from Hyperelasticity.utils import Common
from Hyperelasticity.utils import Evaluate
from Hyperelasticity.utils import MechanicalTests


class Schiavone2009(HyperelasticModel):

    c10 = symbols('c10')
    c30 = symbols('c30')

    model = 'Nonlinear'
    environment = 'In vivo'
    tissue = 'Human'


    def __init__(self, name, _c10, _c30):
        super().__init__(name)
        self._c10 = _c10
        self._c30 = _c30


    def StrainEnergyDensityFunction(self):
        # super().StrainEnergyDensityFunction(self.Ic, self.IIc, self.IIIc)

        print(self.name)

        self.W = self.c10 * (self.Ic - 3) + self.c30 * pow(self.Ic - 3, 3)
        print('W = ')
        print(self.W)
        # print(expand(W))
        # print(factor(W))


    def DefineFunctionsInStretches(self, _testFunc):
        # Mechanical test
        l, l_l1, l_l2, l_l3 = _testFunc()  # e.g. UniaxialTest

        self.Substitute(self.c10, self._c10)
        self.Substitute(self.c30, self._c30)
        self.Substitute(self.l1, l_l1)
        self.Substitute(self.l2, l_l2)
        self.Substitute(self.l3, l_l3)

        return l

