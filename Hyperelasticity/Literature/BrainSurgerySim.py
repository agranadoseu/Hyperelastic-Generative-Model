import numpy as np

from Hyperelasticity.utils import Common
from Hyperelasticity.utils import PlotMaterial
from Hyperelasticity.utils import MechanicalTests
from Hyperelasticity import Linear, Nonlinear

kPa = Common.kPa
p = Common.nP
s = Common.sample
c = 0

"""
.........................................................
BRAIN SURGERY SIMULATION
.........................................................
"""
def Process():

    # output
    data_l = np.zeros((1, s))
    data_W = np.zeros((1, s))
    data_dW = np.zeros((1, s))
    data_H = np.zeros((1, s))
    data_sigma = np.zeros((1, s))
    data_P = np.zeros((1, p), dtype=bool)
    data_offset = []
    data_Y = [""]
    data_C = []
    data_A = []
    data_M = []

    # [Dequidt2015]
    model = Linear.Generic("Linear brain surgery simulator [Dequidt2015]", 0, 2.1 * kPa, 0.45)
    model.Compute()
    l, W, dW, H, sigma, P, offset = model.Evaluate(MechanicalTests.Uniaxial, [0.9,1.1])
    data_l[0, :] = l
    data_W[0, :] = W
    data_dW[0, :] = dW[0]
    data_H[0, :] = H[0,0]
    data_sigma[0, :] = sigma[0]
    data_P[0, :] = P
    data_offset.append(offset)
    data_Y[0] = "SurgSim-Dequidt2015"
    data_C.append(c)
    data_A.append(Common.NH)
    data_M.append(Common.LIN)
    stretches = model.stretches
    # model.Plot()

    # [Sase2015]
    model = Linear.Generic("Linear brain surgery simulator [Sase2015]", 0, 1.0 * kPa, 0.4)
    model.Compute()
    l, W, dW, H, sigma, P, offset = model.Evaluate(MechanicalTests.Uniaxial, [0.9,1.1])
    data_l = np.concatenate((data_l, l.reshape(1, s)), axis=0)
    data_W = np.concatenate((data_W, W.reshape(1, s)), axis=0)
    data_dW = np.concatenate((data_dW, dW[0].reshape(1, s)), axis=0)
    data_H = np.concatenate((data_H, H[0,0].reshape(1, s)), axis=0)
    data_sigma = np.concatenate((data_sigma, sigma[0].reshape(1, s)), axis=0)
    data_P = np.concatenate((data_P, P.reshape(1, p)), axis=0)
    data_offset.append(offset)
    data_Y.append("SurgSim-Sase2015")
    data_C.append(c)
    data_A.append(Common.NH)
    data_M.append(Common.LIN)
    # model.Plot()

    PlotMaterial.Groups("Brain Surgery Simulation", stretches, data_l, data_W, data_dW, data_H, data_sigma, data_Y)

    return data_l, data_W, data_dW, data_H, data_sigma, data_P, data_offset, data_Y, data_C, data_A, data_M
