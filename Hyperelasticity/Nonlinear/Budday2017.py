"""
[Budday2017]
"""

import numpy as np
from sympy import *

from Hyperelasticity.HyperelasticModel import HyperelasticModel
from Hyperelasticity.utils import Common
from Hyperelasticity.utils import Evaluate
from Hyperelasticity.utils import Postulates
from Hyperelasticity.utils import MechanicalTests

s = Common.sample

''' Neo-Hookean '''
class Budday2017a(HyperelasticModel):

    mu = symbols('mu')

    model = 'Nonlinear'
    environment = 'Ex vivo'
    tissue = 'Human'


    def __init__(self, name, _muC, _muT):
        super().__init__(name)
        self._muC = _muC
        self._muT = _muT

    def StrainEnergyDensityFunction(self):
        print(self.name)

        self.W = self.mu / 2.0 * (self.l1**2 + self.l2**2 + self.l3**2 - 3)
        print('W = ')
        print(self.W)

    def DefineFunctionsInStretches(self, _testFunc):
        # Mechanical test
        l, l_l1, l_l2, l_l3 = _testFunc()  # e.g. UniaxialTest

        self.Substitute(self.mu, self._muC)
        self.Substitute(self.l1, l_l1)
        self.Substitute(self.l2, l_l2)
        self.Substitute(self.l3, l_l3)

        return l

    def Evaluate(self, mechTest, stretchBound):
        ''' All '''
        self._W = self.W
        self._dW = self.dW
        self._H = self.H
        self._sigma = self.sigma

        l, l_l1, l_l2, l_l3 = MechanicalTests.Uniaxial()
        self.Substitute(self.mu, self._muC)
        self.Substitute(self.l1, l_l1)
        self.Substitute(self.l2, l_l2)
        self.Substitute(self.l3, l_l3)

        self.stretches = np.linspace(stretchBound[0], stretchBound[1], s)
        self.W_v, self.dW_v, self.H_v, self.sigma_v = Evaluate.Functions(l, self.stretches, self._W, self._dW, self._H, self._sigma)

        ''' Compression '''
        self._W = self.W
        self._dW = self.dW
        self._H = self.H
        self._sigma = self.sigma

        l, l_l1, l_l2, l_l3 = MechanicalTests.Uniaxial()
        self.Substitute(self.mu, self._muC)
        self.Substitute(self.l1, l_l1)
        self.Substitute(self.l2, l_l2)
        self.Substitute(self.l3, l_l3)

        stretchesC = np.linspace(stretchBound[0], 1.0, s/2)
        # W_vC, dW_vC, H_vC, sigma_vC = Evaluate.Functions(l, stretchesC, self._W, self._dW, self._H, self._sigma)
        f_W = lambdify(l, self._W, "numpy")
        W_vC = f_W(stretchesC)
        # print("stretchesC",stretchesC.shape)
        # print("W_vC",W_vC.shape)

        ''' Tension '''
        self._W = self.W
        self._dW = self.dW
        self._H = self.H
        self._sigma = self.sigma

        l, l_l1, l_l2, l_l3 = MechanicalTests.Uniaxial()
        self.Substitute(self.mu, self._muT)
        self.Substitute(self.l1, l_l1)
        self.Substitute(self.l2, l_l2)
        self.Substitute(self.l3, l_l3)

        stretchesT = np.linspace(1.0, stretchBound[1], s / 2)
        # W_vT, dW_vT, H_vT, sigma_vT = Evaluate.Functions(l, stretchesT, self._W, self._dW, self._H, self._sigma)
        f_W = lambdify(l, self._W, "numpy")
        W_vT = f_W(stretchesT)
        # print("stretchesT",stretchesT.shape)
        # print("W_vT",W_vT.shape)

        # TODO Postulates (CORRECT since these is a combined tension compression test)
        P, dWoffset = Postulates.verify(l, self.stretches, self._W, self._dW, self._H, self._sigma)

        ''' Join tests (W and stretches for now) '''
        self.stretches = np.append(stretchesC, stretchesT)
        self.W_v = np.append(W_vC, W_vT)
        # print("stretches", self.stretches.shape)
        # print("W_v", self.W_v.shape)

        return self.stretches, self.W_v, self.dW_v, self.H_v, self.sigma_v, P, dWoffset





''' Mooney-Rivlin '''
class Budday2017b(HyperelasticModel):

    mu = symbols('mu')
    c2 = symbols('c2')

    model = 'Nonlinear'
    environment = 'Ex vivo'
    tissue = 'Human'


    def __init__(self, name, _muC, _c2C, _muT, _c2T):
        super().__init__(name)
        self._muC = _muC
        self._c2C = _c2C
        self._muT = _muT
        self._c2T = _c2T


    def StrainEnergyDensityFunction(self):
        print(self.name)

        self.W = (self.mu/2. - self.c2) * (self.l1**2 + self.l2**2 + self.l3**2 - 3) + \
                 self.c2 * (self.l1**(-2) + self.l2**(-2) + self.l3**(-2) - 3)
        print('W = ')
        print(self.W)


    def DefineFunctionsInStretches(self, _testFunc):
        # Mechanical test
        l, l_l1, l_l2, l_l3 = _testFunc()  # e.g. UniaxialTest

        self.Substitute(self.mu, self._muC)
        self.Substitute(self.c2, self._c2C)
        self.Substitute(self.l1, l_l1)
        self.Substitute(self.l2, l_l2)
        self.Substitute(self.l3, l_l3)

        return l

    def Evaluate(self, mechTest, stretchBound):
        ''' All '''
        self._W = self.W
        self._dW = self.dW
        self._H = self.H
        self._sigma = self.sigma

        l, l_l1, l_l2, l_l3 = MechanicalTests.Uniaxial()
        self.Substitute(self.mu, self._muC)
        self.Substitute(self.c2, self._c2C)
        self.Substitute(self.l1, l_l1)
        self.Substitute(self.l2, l_l2)
        self.Substitute(self.l3, l_l3)

        self.stretches = np.linspace(stretchBound[0], stretchBound[1], s)
        self.W_v, self.dW_v, self.H_v, self.sigma_v = Evaluate.Functions(l, self.stretches, self._W, self._dW, self._H, self._sigma)

        ''' Compression '''
        self._W = self.W
        self._dW = self.dW
        self._H = self.H
        self._sigma = self.sigma

        l, l_l1, l_l2, l_l3 = MechanicalTests.Uniaxial()
        self.Substitute(self.mu, self._muC)
        self.Substitute(self.c2, self._c2C)
        self.Substitute(self.l1, l_l1)
        self.Substitute(self.l2, l_l2)
        self.Substitute(self.l3, l_l3)

        stretchesC = np.linspace(stretchBound[0], 1.0, s/2)
        # W_vC, dW_vC, H_vC, sigma_vC = Evaluate.Functions(l, stretchesC, self._W, self._dW, self._H, self._sigma)
        f_W = lambdify(l, self._W, "numpy")
        W_vC = f_W(stretchesC)
        # print("stretchesC",stretchesC.shape)
        # print("W_vC",W_vC.shape)

        ''' Tension '''
        self._W = self.W
        self._dW = self.dW
        self._H = self.H
        self._sigma = self.sigma

        l, l_l1, l_l2, l_l3 = MechanicalTests.Uniaxial()
        self.Substitute(self.mu, self._muT)
        self.Substitute(self.c2, self._c2T)
        self.Substitute(self.l1, l_l1)
        self.Substitute(self.l2, l_l2)
        self.Substitute(self.l3, l_l3)

        stretchesT = np.linspace(1.0, stretchBound[1], s / 2)
        # W_vT, dW_vT, H_vT, sigma_vT = Evaluate.Functions(l, stretchesT, self._W, self._dW, self._H, self._sigma)
        f_W = lambdify(l, self._W, "numpy")
        W_vT = f_W(stretchesT)
        # print("stretchesT",stretchesT.shape)
        # print("W_vT",W_vT.shape)

        # TODO Postulates (CORRECT since these is a combined tension compression test)
        P, dWoffset = Postulates.verify(l, self.stretches, self._W, self._dW, self._H, self._sigma)

        ''' Join tests (W and stretches for now) '''
        self.stretches = np.append(stretchesC, stretchesT)
        self.W_v = np.append(W_vC, W_vT)
        # print("stretches", self.stretches.shape)
        # print("W_v", self.W_v.shape)

        return self.stretches, self.W_v, self.dW_v, self.H_v, self.sigma_v, P, dWoffset





''' Demiray '''
class Budday2017c(HyperelasticModel):

    mu = symbols('mu')
    beta = symbols('beta')

    model = 'Nonlinear'
    environment = 'Ex vivo'
    tissue = 'Human'


    def __init__(self, name, _muC, _betaC, _muT, _betaT):
        super().__init__(name)
        self._muC = _muC
        self._betaC = _betaC
        self._muT = _muT
        self._betaT = _betaT


    def StrainEnergyDensityFunction(self):
        print(self.name)

        self.W = self.mu/(2.*self.beta) * (functions.elementary.exponential.exp(self.beta*(self.l1**2 + self.l2**2 + self.l3**2 - 3)) -1)
        print('W = ')
        print(self.W)


    def DefineFunctionsInStretches(self, _testFunc):
        # Mechanical test
        l, l_l1, l_l2, l_l3 = _testFunc()  # e.g. UniaxialTest

        self.Substitute(self.mu, self._muC)
        self.Substitute(self.beta, self._betaC)
        self.Substitute(self.l1, l_l1)
        self.Substitute(self.l2, l_l2)
        self.Substitute(self.l3, l_l3)

        return l

    def Evaluate(self, mechTest, stretchBound):
        ''' All '''
        self._W = self.W
        self._dW = self.dW
        self._H = self.H
        self._sigma = self.sigma

        l, l_l1, l_l2, l_l3 = MechanicalTests.Uniaxial()
        self.Substitute(self.mu, self._muC)
        self.Substitute(self.beta, self._betaC)
        self.Substitute(self.l1, l_l1)
        self.Substitute(self.l2, l_l2)
        self.Substitute(self.l3, l_l3)

        self.stretches = np.linspace(stretchBound[0], stretchBound[1], s)
        self.W_v, self.dW_v, self.H_v, self.sigma_v = Evaluate.Functions(l, self.stretches, self._W, self._dW, self._H, self._sigma)

        ''' Compression '''
        self._W = self.W
        self._dW = self.dW
        self._H = self.H
        self._sigma = self.sigma

        l, l_l1, l_l2, l_l3 = MechanicalTests.Uniaxial()
        self.Substitute(self.mu, self._muC)
        self.Substitute(self.beta, self._betaC)
        self.Substitute(self.l1, l_l1)
        self.Substitute(self.l2, l_l2)
        self.Substitute(self.l3, l_l3)

        stretchesC = np.linspace(stretchBound[0], 1.0, s/2)
        # W_vC, dW_vC, H_vC, sigma_vC = Evaluate.Functions(l, stretchesC, self._W, self._dW, self._H, self._sigma)
        f_W = lambdify(l, self._W, "numpy")
        W_vC = f_W(stretchesC)
        # print("stretchesC",stretchesC.shape)
        # print("W_vC",W_vC.shape)

        ''' Tension '''
        self._W = self.W
        self._dW = self.dW
        self._H = self.H
        self._sigma = self.sigma

        l, l_l1, l_l2, l_l3 = MechanicalTests.Uniaxial()
        self.Substitute(self.mu, self._muT)
        self.Substitute(self.beta, self._betaT)
        self.Substitute(self.l1, l_l1)
        self.Substitute(self.l2, l_l2)
        self.Substitute(self.l3, l_l3)

        stretchesT = np.linspace(1.0, stretchBound[1], s / 2)
        # W_vT, dW_vT, H_vT, sigma_vT = Evaluate.Functions(l, stretchesT, self._W, self._dW, self._H, self._sigma)
        f_W = lambdify(l, self._W, "numpy")
        W_vT = f_W(stretchesT)
        # print("stretchesT",stretchesT.shape)
        # print("W_vT",W_vT.shape)

        # TODO Postulates (CORRECT since these is a combined tension compression test)
        P, dWoffset = Postulates.verify(l, self.stretches, self._W, self._dW, self._H, self._sigma)

        ''' Join tests (W and stretches for now) '''
        self.stretches = np.append(stretchesC, stretchesT)
        self.W_v = np.append(W_vC, W_vT)
        # print("stretches", self.stretches.shape)
        # print("W_v", self.W_v.shape)

        return self.stretches, self.W_v, self.dW_v, self.H_v, self.sigma_v, P, dWoffset




''' Gent '''
class Budday2017d(HyperelasticModel):

    mu = symbols('mu')
    eta = symbols('eta')

    model = 'Nonlinear'
    environment = 'Ex vivo'
    tissue = 'Human'


    def __init__(self, name, _muC, _etaC, _muT, _etaT):
        super().__init__(name)
        self._muC = _muC
        self._etaC = _etaC
        self._muT = _muT
        self._etaT = _etaT


    def StrainEnergyDensityFunction(self):
        print(self.name)

        self.W = -self.mu*self.eta/2. * functions.elementary.exponential.log(1 - (self.l1**2 + self.l2**2 + self.l3**2 - 3)/self.eta)
        print('W = ')
        print(self.W)


    def DefineFunctionsInStretches(self, _testFunc):
        # Mechanical test
        l, l_l1, l_l2, l_l3 = _testFunc()  # e.g. UniaxialTest

        self.Substitute(self.mu, self._muC)
        self.Substitute(self.eta, self._etaC)
        self.Substitute(self.l1, l_l1)
        self.Substitute(self.l2, l_l2)
        self.Substitute(self.l3, l_l3)

        return l

    def Evaluate(self, mechTest, stretchBound):
        ''' All '''
        self._W = self.W
        self._dW = self.dW
        self._H = self.H
        self._sigma = self.sigma

        l, l_l1, l_l2, l_l3 = MechanicalTests.Uniaxial()
        self.Substitute(self.mu, self._muC)
        self.Substitute(self.eta, self._etaC)
        self.Substitute(self.l1, l_l1)
        self.Substitute(self.l2, l_l2)
        self.Substitute(self.l3, l_l3)

        self.stretches = np.linspace(stretchBound[0], stretchBound[1], s)
        self.W_v, self.dW_v, self.H_v, self.sigma_v = Evaluate.Functions(l, self.stretches, self._W, self._dW, self._H, self._sigma)

        ''' Compression '''
        self._W = self.W
        self._dW = self.dW
        self._H = self.H
        self._sigma = self.sigma

        l, l_l1, l_l2, l_l3 = MechanicalTests.Uniaxial()
        self.Substitute(self.mu, self._muC)
        self.Substitute(self.eta, self._etaC)
        self.Substitute(self.l1, l_l1)
        self.Substitute(self.l2, l_l2)
        self.Substitute(self.l3, l_l3)

        stretchesC = np.linspace(stretchBound[0], 1.0, s/2)
        # W_vC, dW_vC, H_vC, sigma_vC = Evaluate.Functions(l, stretchesC, self._W, self._dW, self._H, self._sigma)
        f_W = lambdify(l, self._W, "numpy")
        W_vC = f_W(stretchesC)
        # print("stretchesC",stretchesC.shape)
        # print("W_vC",W_vC.shape)

        ''' Tension '''
        self._W = self.W
        self._dW = self.dW
        self._H = self.H
        self._sigma = self.sigma

        l, l_l1, l_l2, l_l3 = MechanicalTests.Uniaxial()
        self.Substitute(self.mu, self._muT)
        self.Substitute(self.eta, self._etaT)
        self.Substitute(self.l1, l_l1)
        self.Substitute(self.l2, l_l2)
        self.Substitute(self.l3, l_l3)

        stretchesT = np.linspace(1.0, stretchBound[1], s / 2)
        # W_vT, dW_vT, H_vT, sigma_vT = Evaluate.Functions(l, stretchesT, self._W, self._dW, self._H, self._sigma)
        f_W = lambdify(l, self._W, "numpy")
        W_vT = f_W(stretchesT)
        # print("stretchesT",stretchesT.shape)
        # print("W_vT",W_vT.shape)

        # TODO Postulates (CORRECT since these is a combined tension compression test)
        P, dWoffset = Postulates.verify(l, self.stretches, self._W, self._dW, self._H, self._sigma)

        ''' Join tests (W and stretches for now) '''
        self.stretches = np.append(stretchesC, stretchesT)
        self.W_v = np.append(W_vC, W_vT)
        # print("stretches", self.stretches.shape)
        # print("W_v", self.W_v.shape)

        return self.stretches, self.W_v, self.dW_v, self.H_v, self.sigma_v, P, dWoffset




''' Modified 1-term Ogden '''
class Budday2017e(HyperelasticModel):

    mu = symbols('mu')
    alpha = symbols('alpha')

    model = 'Nonlinear'
    environment = 'Ex vivo'
    tissue = 'Human'


    def __init__(self, name, _muC, _alphaC, _muT, _alphaT):
        super().__init__(name)
        self._muC = _muC
        self._alphaC = _alphaC
        self._muT = _muT
        self._alphaT = _alphaT


    def StrainEnergyDensityFunction(self):
        print(self.name)

        self.W = 2.*self.mu / (self.alpha**2) * (self.l1**self.alpha + self.l2**self.alpha + self.l3**self.alpha - 3)
        print('W = ')
        print(self.W)

    def DefineFunctionsInStretches(self, _testFunc):
        # Mechanical test
        l, l_l1, l_l2, l_l3 = _testFunc()  # e.g. UniaxialTest

        self.Substitute(self.mu, self._muC)
        self.Substitute(self.alpha, self._alphaC)
        self.Substitute(self.l1, l_l1)
        self.Substitute(self.l2, l_l2)
        self.Substitute(self.l3, l_l3)

        return l

    def Evaluate(self, mechTest, stretchBound):
        ''' All '''
        self._W = self.W
        self._dW = self.dW
        self._H = self.H
        self._sigma = self.sigma

        l, l_l1, l_l2, l_l3 = MechanicalTests.Uniaxial()
        self.Substitute(self.mu, self._muC)
        self.Substitute(self.alpha, self._alphaC)
        self.Substitute(self.l1, l_l1)
        self.Substitute(self.l2, l_l2)
        self.Substitute(self.l3, l_l3)

        self.stretches = np.linspace(stretchBound[0], stretchBound[1], s)
        self.W_v, self.dW_v, self.H_v, self.sigma_v = Evaluate.Functions(l, self.stretches, self._W, self._dW, self._H, self._sigma)

        ''' Compression '''
        self._W = self.W
        self._dW = self.dW
        self._H = self.H
        self._sigma = self.sigma

        l, l_l1, l_l2, l_l3 = MechanicalTests.Uniaxial()
        self.Substitute(self.mu, self._muC)
        self.Substitute(self.alpha, self._alphaC)
        self.Substitute(self.l1, l_l1)
        self.Substitute(self.l2, l_l2)
        self.Substitute(self.l3, l_l3)

        stretchesC = np.linspace(stretchBound[0], 1.0, s/2)
        # W_vC, dW_vC, H_vC, sigma_vC = Evaluate.Functions(l, stretchesC, self._W, self._dW, self._H, self._sigma)
        f_W = lambdify(l, self._W, "numpy")
        W_vC = f_W(stretchesC)
        # print("stretchesC",stretchesC.shape)
        # print("W_vC",W_vC.shape)

        ''' Tension '''
        self._W = self.W
        self._dW = self.dW
        self._H = self.H
        self._sigma = self.sigma

        l, l_l1, l_l2, l_l3 = MechanicalTests.Uniaxial()
        self.Substitute(self.mu, self._muT)
        self.Substitute(self.alpha, self._alphaT)
        self.Substitute(self.l1, l_l1)
        self.Substitute(self.l2, l_l2)
        self.Substitute(self.l3, l_l3)

        stretchesT = np.linspace(1.0, stretchBound[1], s / 2)
        # W_vT, dW_vT, H_vT, sigma_vT = Evaluate.Functions(l, stretchesT, self._W, self._dW, self._H, self._sigma)
        f_W = lambdify(l, self._W, "numpy")
        W_vT = f_W(stretchesT)
        # print("stretchesT",stretchesT.shape)
        # print("W_vT",W_vT.shape)

        # TODO Postulates (CORRECT since these is a combined tension compression test)
        P, dWoffset = Postulates.verify(l, self.stretches, self._W, self._dW, self._H, self._sigma)

        ''' Join tests (W and stretches for now) '''
        self.stretches = np.append(stretchesC, stretchesT)
        self.W_v = np.append(W_vC, W_vT)
        # print("stretches", self.stretches.shape)
        # print("W_v", self.W_v.shape)

        return self.stretches, self.W_v, self.dW_v, self.H_v, self.sigma_v, P, dWoffset