import numpy as np

from Hyperelasticity.utils import Common
from Hyperelasticity.utils import PlotMaterial
from Hyperelasticity.utils import MechanicalTests
from Hyperelasticity import Linear, Nonlinear

kPa = Common.kPa
p = Common.nP
s = Common.sample
c = 5

"""
.........................................................
LINEAR MRE
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

    # [Kruse1999] add individual cases in addition to this average
    # age =   [18,  20,   26,   45,  52,   52,   62,   79]
    # WM: G = [6.5, 26.5, 12.0, 8.5, 12.0, 18.0, 16.0, 14.0]
    # GM: G = [5.0, 12.0, 5.5,  5.0, 6.0,  7.0,  7.5,  3.0]
    model = Linear.Generic("Linear MRE WM [Kruse1999]", 14.6 * kPa, 42.34 * kPa, 0.45)
    model.Compute()
    l, W, dW, H, sigma, P, offset = model.Evaluate(MechanicalTests.Uniaxial, [0.95,1.05])
    data_l[0, :] = l
    data_W[0, :] = W
    data_dW[0, :] = dW[0]
    data_H[0, :] = H[0,0]
    data_sigma[0, :] = sigma[0]
    data_P[0, :] = P
    data_offset.append(offset)
    data_Y[0] = "MRE-WM-Kruse1999"
    data_C.append(c)
    data_A.append(Common.WM)
    data_M.append(Common.MRE)
    stretches = model.stretches

    model = Linear.Generic("Linear MRE GM [Kruse1999]", 6.43 * kPa, 18.65 * kPa, 0.45)
    model.Compute()
    l, W, dW, H, sigma, P, offset = model.Evaluate(MechanicalTests.Uniaxial, [0.95,1.05])
    data_l = np.concatenate((data_l, l.reshape(1, s)), axis=0)
    data_W = np.concatenate((data_W, W.reshape(1, s)), axis=0)
    data_dW = np.concatenate((data_dW, dW[0].reshape(1, s)), axis=0)
    data_H = np.concatenate((data_H, H[0,0].reshape(1, s)), axis=0)
    data_sigma = np.concatenate((data_sigma, sigma[0].reshape(1, s)), axis=0)
    data_P = np.concatenate((data_P, P.reshape(1, p)), axis=0)
    data_offset.append(offset)
    data_Y.append("MRE-GM-Kruse1999")
    data_C.append(c)
    data_A.append(Common.GM)
    data_M.append(Common.MRE)

    # [Uffmann2004] Need to add models under standard deviation
    model = Linear.Generic("Linear MRE WM [Uffmann2004]", 15.2 * kPa, 44.08 * kPa, 0.45)
    model.Compute()
    l, W, dW, H, sigma, P, offset = model.Evaluate(MechanicalTests.Uniaxial, [0.95,1.05])
    data_l = np.concatenate((data_l, l.reshape(1, s)), axis=0)
    data_W = np.concatenate((data_W, W.reshape(1, s)), axis=0)
    data_dW = np.concatenate((data_dW, dW[0].reshape(1, s)), axis=0)
    data_H = np.concatenate((data_H, H[0, 0].reshape(1, s)), axis=0)
    data_sigma = np.concatenate((data_sigma, sigma[0].reshape(1, s)), axis=0)
    data_P = np.concatenate((data_P, P.reshape(1, p)), axis=0)
    data_offset.append(offset)
    data_Y.append("MRE-WM-Uffmann2004")
    data_C.append(c)
    data_A.append(Common.WM)
    data_M.append(Common.MRE)

    model = Linear.Generic("Linear MRE GM [Uffmann2004]", 12.9 * kPa, 37.41 * kPa, 0.45)
    model.Compute()
    l, W, dW, H, sigma, P, offset = model.Evaluate(MechanicalTests.Uniaxial, [0.95,1.05])
    data_l = np.concatenate((data_l, l.reshape(1, s)), axis=0)
    data_W = np.concatenate((data_W, W.reshape(1, s)), axis=0)
    data_dW = np.concatenate((data_dW, dW[0].reshape(1, s)), axis=0)
    data_H = np.concatenate((data_H, H[0, 0].reshape(1, s)), axis=0)
    data_sigma = np.concatenate((data_sigma, sigma[0].reshape(1, s)), axis=0)
    data_P = np.concatenate((data_P, P.reshape(1, p)), axis=0)
    data_offset.append(offset)
    data_Y.append("MRE-GM-Uffmann2004")
    data_C.append(c)
    data_A.append(Common.GM)
    data_M.append(Common.MRE)

    # [McCracken2005] Need to add models under standard deviation
    model = Linear.Generic("Linear MRE WM [McCracken2005]", 10.7 * kPa, 31.03 * kPa, 0.45)
    model.Compute()
    l, W, dW, H, sigma, P, offset = model.Evaluate(MechanicalTests.Uniaxial, [0.95,1.05])
    data_l = np.concatenate((data_l, l.reshape(1, s)), axis=0)
    data_W = np.concatenate((data_W, W.reshape(1, s)), axis=0)
    data_dW = np.concatenate((data_dW, dW[0].reshape(1, s)), axis=0)
    data_H = np.concatenate((data_H, H[0, 0].reshape(1, s)), axis=0)
    data_sigma = np.concatenate((data_sigma, sigma[0].reshape(1, s)), axis=0)
    data_P = np.concatenate((data_P, P.reshape(1, p)), axis=0)
    data_offset.append(offset)
    data_Y.append("MRE-WM-McCracken2005")
    data_C.append(c)
    data_A.append(Common.WM)
    data_M.append(Common.MRE)

    model = Linear.Generic("Linear MRE GM [McCracken2005]", 5.3 * kPa, 15.37 * kPa, 0.45)
    model.Compute()
    l, W, dW, H, sigma, P, offset = model.Evaluate(MechanicalTests.Uniaxial, [0.95,1.05])
    data_l = np.concatenate((data_l, l.reshape(1, s)), axis=0)
    data_W = np.concatenate((data_W, W.reshape(1, s)), axis=0)
    data_dW = np.concatenate((data_dW, dW[0].reshape(1, s)), axis=0)
    data_H = np.concatenate((data_H, H[0, 0].reshape(1, s)), axis=0)
    data_sigma = np.concatenate((data_sigma, sigma[0].reshape(1, s)), axis=0)
    data_P = np.concatenate((data_P, P.reshape(1, p)), axis=0)
    data_offset.append(offset)
    data_Y.append("MRE-GM-McCracken2005")
    data_C.append(c)
    data_A.append(Common.GM)
    data_M.append(Common.MRE)

    # [Green2006]
    model = Linear.Generic("Linear MRE WM [Green2006]", 2.1 * kPa, 6.09 * kPa, 0.45)
    model.Compute()
    l, W, dW, H, sigma, P, offset = model.Evaluate(MechanicalTests.Uniaxial, [0.95,1.05])
    data_l = np.concatenate((data_l, l.reshape(1, s)), axis=0)
    data_W = np.concatenate((data_W, W.reshape(1, s)), axis=0)
    data_dW = np.concatenate((data_dW, dW[0].reshape(1, s)), axis=0)
    data_H = np.concatenate((data_H, H[0, 0].reshape(1, s)), axis=0)
    data_sigma = np.concatenate((data_sigma, sigma[0].reshape(1, s)), axis=0)
    data_P = np.concatenate((data_P, P.reshape(1, p)), axis=0)
    data_offset.append(offset)
    data_Y.append("MRE-WM-Green2006")
    data_C.append(c)
    data_A.append(Common.WM)
    data_M.append(Common.MRE)

    model = Linear.Generic("Linear MRE GM [Green2006]", 2.8 * kPa, 8.12 * kPa, 0.45)
    model.Compute()
    l, W, dW, H, sigma, P, offset = model.Evaluate(MechanicalTests.Uniaxial, [0.95,1.05])
    data_l = np.concatenate((data_l, l.reshape(1, s)), axis=0)
    data_W = np.concatenate((data_W, W.reshape(1, s)), axis=0)
    data_dW = np.concatenate((data_dW, dW[0].reshape(1, s)), axis=0)
    data_H = np.concatenate((data_H, H[0, 0].reshape(1, s)), axis=0)
    data_sigma = np.concatenate((data_sigma, sigma[0].reshape(1, s)), axis=0)
    data_P = np.concatenate((data_P, P.reshape(1, p)), axis=0)
    data_offset.append(offset)
    data_Y.append("MRE-GM-Green2006")
    data_C.append(c)
    data_A.append(Common.GM)
    data_M.append(Common.MRE)

    # [Kruse2008]
    # TODO CI=[12.3,14.8]
    model = Linear.Generic("Linear MRE WM [Kruse2008]", 13.6 * kPa, 39.44 * kPa, 0.45)
    model.Compute()
    l, W, dW, H, sigma, P, offset = model.Evaluate(MechanicalTests.Uniaxial, [0.95,1.05])
    data_l = np.concatenate((data_l, l.reshape(1, s)), axis=0)
    data_W = np.concatenate((data_W, W.reshape(1, s)), axis=0)
    data_dW = np.concatenate((data_dW, dW[0].reshape(1, s)), axis=0)
    data_H = np.concatenate((data_H, H[0, 0].reshape(1, s)), axis=0)
    data_sigma = np.concatenate((data_sigma, sigma[0].reshape(1, s)), axis=0)
    data_P = np.concatenate((data_P, P.reshape(1, p)), axis=0)
    data_offset.append(offset)
    data_Y.append("MRE-WM-Kruse2008")
    data_C.append(c)
    data_A.append(Common.WM)
    data_M.append(Common.MRE)

    # TODO This case was added to include a MRE abnormality (100 times G)
    model = Linear.Generic("Linear MRE abnormal WM [Kruse2008]", 1360 * kPa, 3944 * kPa, 0.45)
    model.Compute()
    l, W, dW, H, sigma, P, offset = model.Evaluate(MechanicalTests.Uniaxial, [0.95, 1.05])
    data_l = np.concatenate((data_l, l.reshape(1, s)), axis=0)
    data_W = np.concatenate((data_W, W.reshape(1, s)), axis=0)
    data_dW = np.concatenate((data_dW, dW[0].reshape(1, s)), axis=0)
    data_H = np.concatenate((data_H, H[0, 0].reshape(1, s)), axis=0)
    data_sigma = np.concatenate((data_sigma, sigma[0].reshape(1, s)), axis=0)
    data_P = np.concatenate((data_P, P.reshape(1, p)), axis=0)
    data_offset.append(offset)
    data_Y.append("MRE-abnormal-WM-Kruse2008")
    data_C.append(c)
    data_A.append(Common.AB)
    data_M.append(Common.MRE)

    # TODO CI=[4.76,5.66]
    model = Linear.Generic("Linear MRE GM [Kruse2008]", 5.22 * kPa, 15.14 * kPa, 0.45)
    model.Compute()
    l, W, dW, H, sigma, P, offset = model.Evaluate(MechanicalTests.Uniaxial, [0.95,1.05])
    data_l = np.concatenate((data_l, l.reshape(1, s)), axis=0)
    data_W = np.concatenate((data_W, W.reshape(1, s)), axis=0)
    data_dW = np.concatenate((data_dW, dW[0].reshape(1, s)), axis=0)
    data_H = np.concatenate((data_H, H[0, 0].reshape(1, s)), axis=0)
    data_sigma = np.concatenate((data_sigma, sigma[0].reshape(1, s)), axis=0)
    data_P = np.concatenate((data_P, P.reshape(1, p)), axis=0)
    data_offset.append(offset)
    data_Y.append("MRE-GM-Kruse2008")
    data_C.append(c)
    data_A.append(Common.GM)
    data_M.append(Common.MRE)

    # TODO G=[1.7,6.1] over gray and white matter
    model = Linear.Generic("Linear MRE [Hamhaber2007]", 3.5 * kPa, 10.15 * kPa, 0.45)
    model.Compute()
    l, W, dW, H, sigma, P, offset = model.Evaluate(MechanicalTests.Uniaxial, [0.95,1.05])
    data_l = np.concatenate((data_l, l.reshape(1, s)), axis=0)
    data_W = np.concatenate((data_W, W.reshape(1, s)), axis=0)
    data_dW = np.concatenate((data_dW, dW[0].reshape(1, s)), axis=0)
    data_H = np.concatenate((data_H, H[0, 0].reshape(1, s)), axis=0)
    data_sigma = np.concatenate((data_sigma, sigma[0].reshape(1, s)), axis=0)
    data_P = np.concatenate((data_P, P.reshape(1, p)), axis=0)
    data_offset.append(offset)
    data_Y.append("MRE-Hamhaber2007")
    data_C.append(c)
    data_A.append(Common.NH)
    data_M.append(Common.MRE)

    PlotMaterial.Groups("MR Elastography", stretches, data_l, data_W, data_dW, data_H, data_sigma, data_Y)

    return data_l, data_W, data_dW, data_H, data_sigma, data_P, data_offset, data_Y, data_C, data_A, data_M