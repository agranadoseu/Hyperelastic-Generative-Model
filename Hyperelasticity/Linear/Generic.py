"""
[Kruse1999] from [Morin2017]
"""

import numpy as np
from sympy import *

from Hyperelasticity.HyperelasticModel import HyperelasticModel
from Hyperelasticity.utils import Common
from Hyperelasticity.utils import Evaluate
from Hyperelasticity.utils import MechanicalTests


class Generic(HyperelasticModel):

    muLame = symbols('muLame')
    lambdaLame = symbols('lambdaLame')

    model = 'Linear'
    environment = 'In vivo'
    tissue = 'Human'

    def __init__(self, name, _shearModulus, _YoungsModulus, _PoissonsRatio):
        super().__init__(name)

        if _YoungsModulus==0:
            _YoungsModulus = 2 * _shearModulus * (1 + _PoissonsRatio)
        if _shearModulus==0:
            _shearModulus = _YoungsModulus / (2*(1+_PoissonsRatio))

        print(self.name)
        print('Elastic properties: G={:.2f}, E={:.2f}, v={:.2f}'.format(_shearModulus, _YoungsModulus, _PoissonsRatio))

        self._muLame = _YoungsModulus / (2*(1+_PoissonsRatio))
        self._lambdaLame = (_YoungsModulus*_PoissonsRatio) / ((1+_PoissonsRatio)*(1-2*_PoissonsRatio))



    def StrainEnergyDensityFunction(self):

        # small strain tensor: e = 1/2(F+F^T)-I = Fhat-I
        # here assuming only rotational invariant deformation gradient
        self.epsilon1 = self.l1 - 1
        self.epsilon2 = self.l2 - 1
        self.epsilon3 = self.l3 - 1

        # [Sifakis2012] Sec 3.4
        # Method A: Linear / Co-rotational
        # trace(A) = sum(a_ii)
        # trace(A^T * B) = sum(a_ij * b_ij)
        # self.W = self.muLame * Trace(Transpose(self.epsilon) * self.epsilon) + self.lambdaLame / 2 * Trace(self.epsilon)**2
        self.W = self.muLame * (self.epsilon1**2 + self.epsilon2**2 + self.epsilon3**2) + \
                 self.lambdaLame / 2 * (self.epsilon1 + self.epsilon2 + self.epsilon3)**2

        print('W = ')
        print(self.W)
        # print(expand(W))
        # print(factor(W))


    def DefineFunctionsInStretches(self, _testFunc):
        # Mechanical test
        l, l_l1, l_l2, l_l3 = _testFunc()  # e.g. UniaxialTest

        self.Substitute(self.muLame, self._muLame)
        self.Substitute(self.lambdaLame, self._lambdaLame)
        self.Substitute(self.l1, l_l1)
        self.Substitute(self.l2, l_l2)
        self.Substitute(self.l3, l_l3)

        return l


