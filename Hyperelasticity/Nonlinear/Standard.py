"""
Neo-Hookean
Mooney-Rivlin
Ogden
"""

import numpy as np
from sympy import *

from Hyperelasticity.HyperelasticModel import HyperelasticModel
from Hyperelasticity.utils import Common
from Hyperelasticity.utils import Evaluate
from Hyperelasticity.utils import MechanicalTests


class NeoHookean(HyperelasticModel):

    mu = symbols('mu')

    model = 'Nonlinear'
    environment = 'Ex vivo'
    tissue = 'Human'

    def __init__(self, name, _mu):
        super().__init__(name)
        self._mu = _mu

    def StrainEnergyDensityFunction(self):
        print(self.name)

        # c(Ic-3)/2
        # self.W = (self.mu / 2.) * (self.l1**2 + self.l2**2 + self.l3**2 - 3)
        self.W = (self.mu/2.) * (self.l1**2 + 2.*self.l1**(-1.) - 3)
        # print('W = ')
        # print(self.W)
        # print(expand(W))
        # print(factor(W))

    def DefineFunctionsInStretches(self, _testFunc):
        # Mechanical test
        l, l_l1, l_l2, l_l3 = _testFunc()  # e.g. UniaxialTest

        self.Substitute(self.mu, self._mu)
        self.Substitute(self.l1, l_l1)
        self.Substitute(self.l2, l_l2)
        self.Substitute(self.l3, l_l3)

        return l


class MooneyRivlin(HyperelasticModel):

    c1 = symbols('c1')
    c2 = symbols('c2')

    model = 'Nonlinear'
    environment = 'Ex vivo'
    tissue = 'Human'

    def __init__(self, name, _c1, _c2):
        super().__init__(name)
        self._c1 = _c1
        self._c2 = _c2

    def StrainEnergyDensityFunction(self):
        print(self.name)

        # self.W = self.c1 * (self.l1**2 + self.l2**2 + self.l3**2 - 3) + \
        #          self.c2 * (self.l1**2 * self.l2**2 + self.l2**2 * self.l3**2 + self.l3**2 * self.l1**2 - 3)
        self.W = self.c1 * (self.l1**2 + 2.*self.l1**(-1.) - 3) + \
                 self.c2 * (2.*self.l1 + self.l1**(-2.) - 3)
        # print('W = ')
        # print(self.W)
        # print(expand(W))
        # print(factor(W))

    def DefineFunctionsInStretches(self, _testFunc):
        # Mechanical test
        l, l_l1, l_l2, l_l3 = _testFunc()  # e.g. UniaxialTest

        self.Substitute(self.c1, self._c1)
        self.Substitute(self.c2, self._c2)
        self.Substitute(self.l1, l_l1)
        self.Substitute(self.l2, l_l2)
        self.Substitute(self.l3, l_l3)

        return l


class Ogden(HyperelasticModel):

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

        # print(self.cp)
        # print(self.mp)

    def StrainEnergyDensityFunction(self):
        # print(self.name)

        self. W = 0

        for i in range(self._N):
            # self.W += (self.cp[i]/self.mp[i]) * (self.l1**self.mp[i] + self.l2**self.mp[i] + self.l3**self.mp[i] - 3)
            self.W += (self.cp[i]/self.mp[i]) * (self.l1**self.mp[i] + 2.0*self.l1**(-self.mp[i]/2.0) - 3)

        # print('W = ')
        # print(self.W)
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
