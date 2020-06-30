"""
[Mihai2017]
"""

import numpy as np
from sympy import *

from Hyperelasticity.HyperelasticModel import HyperelasticModel
from Hyperelasticity.utils import Common
from Hyperelasticity.utils import Evaluate
from Hyperelasticity.utils import MechanicalTests


class Mihai2017a(HyperelasticModel):

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

        self.W = self.c0 * (self.l1**(2*self.alpha) + self.l2**(2*self.alpha) + self.l3**(2*self.alpha) - 3) / (2*self.alpha)
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






class Mihai2017b(HyperelasticModel):

    c0 = symbols('c0')
    alpha = symbols('alpha')
    c1 = symbols('c1')
    c2 = symbols('c2')

    model = 'Nonlinear'
    environment = 'Ex vivo'
    tissue = 'Human'


    def __init__(self, name, _c0, _alpha, _c1, _c2):
        super().__init__(name)
        self._c0 = _c0
        self._alpha = _alpha
        self._c1 = _c1
        self._c2 = _c2


    def StrainEnergyDensityFunction(self):
        print(self.name)

        self.W = self.c0 * (self.l1**(2*self.alpha) + self.l2**(2*self.alpha) + self.l3**(2*self.alpha) - 3) / (2*self.alpha) + \
                 self.c1 * (self.l1 ** 2 + self.l2 ** 2 + self.l3 ** 2 - 3) / 2 + \
                 self.c2 * (self.l1 ** (-2) + self.l2 ** (-2) + self.l3 ** (-2) - 3) / 2
        print('W = ')
        print(self.W)
        # print(expand(W))
        # print(factor(W))


    def DefineFunctionsInStretches(self, _testFunc):
        # Mechanical test
        l, l_l1, l_l2, l_l3 = _testFunc()  # e.g. UniaxialTest

        self.Substitute(self.c0, self._c0)
        self.Substitute(self.alpha, self._alpha)
        self.Substitute(self.c1, self._c1)
        self.Substitute(self.c2, self._c2)
        self.Substitute(self.l1, l_l1)
        self.Substitute(self.l2, l_l2)
        self.Substitute(self.l3, l_l3)

        return l





class Mihai2017c(HyperelasticModel):

    model = 'Nonlinear'
    environment = 'Ex vivo'
    tissue = 'Human'


    def __init__(self, name, _N, _cp, _mp):
        super().__init__(name)
        self._N = _N
        self._cp = _cp
        self._mp = _mp

        # create names of symbols
        self.strCp = ["" for x in range(self._N)]
        self.strMp = ["" for x in range(self._N)]
        for i in range(self._N):
            # create text array
            self.strCp[i] = 'cp' + str(i)
            self.strMp[i] = 'mp' + str(i)

        # create symbols
        dummy = symbols('dummmy')
        # print(dummy)
        self.cp = [dummy for x in range(self._N)]
        self.mp = [dummy for x in range(self._N)]
        for i in range(self._N):
            self.cp[i] = symbols(self.strCp[i])
            self.mp[i] = symbols(self.strMp[i])

        print(self.cp)
        print(self.mp)


    def StrainEnergyDensityFunction(self):
        print(self.name)

        self. W = 0

        for i in range(self._N):
            self.W += self.cp[i] * (self.l1**(self.mp[i]) + self.l2**(self.mp[i]) + self.l3**(self.mp[i]) - 3) / self.mp[i]
            # self.W += self.cp[i] * (self.l1**(self.mp[i]) - 1) / self.mp[i]

        print('W = ')
        print(self.W)
        # print(expand(W))
        # print(factor(W))


    def DefineFunctionsInStretches(self, _testFunc):
        # Mechanical test
        l, l_l1, l_l2, l_l3 = _testFunc()  # e.g. UniaxialTest

        for i in range(self._N):
            self.Substitute(self.cp[i], self._cp[i])
            self.Substitute(self.mp[i], self._mp[i])
        self.Substitute(self.l1, l_l1)
        self.Substitute(self.l2, l_l2)
        self.Substitute(self.l3, l_l3)

        return l