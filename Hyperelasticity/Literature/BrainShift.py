import numpy as np

from Hyperelasticity.utils import Common
from Hyperelasticity.utils import PlotMaterial
from Hyperelasticity.utils import MechanicalTests
from Hyperelasticity import Linear, Nonlinear

kPa = Common.kPa
p = Common.nP
s = Common.sample
c = 4

"""
.........................................................
LINEAR BRAIN-SHIFT PARKINSON
LINEAR BRAIN SHIFT COMPENSATION FOR ABLATION
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

    # [Clatz2005]
    model = Linear.Generic("Linear brain-shift compensation for ablation [Clatz2005]", 0, 0.694 * kPa, 0.45)
    model.Compute()
    l, W, dW, H, sigma, P, offset = model.Evaluate(MechanicalTests.Uniaxial, [0.9,1.1])
    data_l[0, :] = l
    data_W[0, :] = W
    data_dW[0, :] = dW[0]
    data_H[0, :] = H[0,0]
    data_sigma[0, :] = sigma[0]
    data_P[0, :] = P
    data_offset.append(offset)
    data_Y[0] = "BrainShift-Clatz2005"
    data_C.append(c)
    data_A.append(Common.NH)
    data_M.append(Common.LIN)
    stretches = model.stretches

    # [Wittek2009]
    model = Linear.Generic("Linear brain-shift compensation for ablation [Wittek2009]", 0, 2.5 * kPa, 0.49)
    model.Compute()
    l, W, dW, H, sigma, P, offset = model.Evaluate(MechanicalTests.Uniaxial, [0.9,1.1])
    data_l = np.concatenate((data_l, l.reshape(1, s)), axis=0)
    data_W = np.concatenate((data_W, W.reshape(1, s)), axis=0)
    data_dW = np.concatenate((data_dW, dW[0].reshape(1, s)), axis=0)
    data_H = np.concatenate((data_H, H[0,0].reshape(1, s)), axis=0)
    data_sigma = np.concatenate((data_sigma, sigma[0].reshape(1, s)), axis=0)
    data_P = np.concatenate((data_P, P.reshape(1, p)), axis=0)
    data_offset.append(offset)
    data_Y.append("BrainShift-Wittek2009")
    data_C.append(c)
    data_A.append(Common.NH)
    data_M.append(Common.LIN)

    # [Dumpuri2006,Chen2011]
    model = Linear.Generic("Linear brain-shift compensation for ablation [Dumpuri2006,Chen2011]", 0, 2.1 * kPa, 0.45)
    model.Compute()
    l, W, dW, H, sigma, P, offset = model.Evaluate(MechanicalTests.Uniaxial, [0.9,1.1])
    data_l = np.concatenate((data_l, l.reshape(1, s)), axis=0)
    data_W = np.concatenate((data_W, W.reshape(1, s)), axis=0)
    data_dW = np.concatenate((data_dW, dW[0].reshape(1, s)), axis=0)
    data_H = np.concatenate((data_H, H[0, 0].reshape(1, s)), axis=0)
    data_sigma = np.concatenate((data_sigma, sigma[0].reshape(1, s)), axis=0)
    data_P = np.concatenate((data_P, P.reshape(1, p)), axis=0)
    data_offset.append(offset)
    data_Y.append("BrainShift-Dumpuri2006")
    data_C.append(c)
    data_A.append(Common.NH)
    data_M.append(Common.LIN)

    # [Vigneron2012]
    model = Linear.Generic("Linear brain-shift compensation for ablation [Vigneron2012]", 0, 3.0 * kPa, 0.45)
    model.Compute()
    l, W, dW, H, sigma, P, offset = model.Evaluate(MechanicalTests.Uniaxial, [0.9,1.1])
    data_l = np.concatenate((data_l, l.reshape(1, s)), axis=0)
    data_W = np.concatenate((data_W, W.reshape(1, s)), axis=0)
    data_dW = np.concatenate((data_dW, dW[0].reshape(1, s)), axis=0)
    data_H = np.concatenate((data_H, H[0, 0].reshape(1, s)), axis=0)
    data_sigma = np.concatenate((data_sigma, sigma[0].reshape(1, s)), axis=0)
    data_Y.append("BrainShift-Vigneron2012")
    data_C.append(c)
    data_A.append(Common.NH)
    data_M.append(Common.LIN)

    # [deLorenzo2012]
    # TODO E too high (may remove, no reference 27 within paper)
    model = Linear.Generic("Linear brain-shift compensation for ablation [deLorenzo2012]", 0, 66.7 * kPa, 0.48)
    model.Compute()
    l, W, dW, H, sigma, P, offset = model.Evaluate(MechanicalTests.Uniaxial, [0.9,1.1])
    data_l = np.concatenate((data_l, l.reshape(1, s)), axis=0)
    data_W = np.concatenate((data_W, W.reshape(1, s)), axis=0)
    data_dW = np.concatenate((data_dW, dW[0].reshape(1, s)), axis=0)
    data_H = np.concatenate((data_H, H[0, 0].reshape(1, s)), axis=0)
    data_sigma = np.concatenate((data_sigma, sigma[0].reshape(1, s)), axis=0)
    data_P = np.concatenate((data_P, P.reshape(1, p)), axis=0)
    data_offset.append(offset)
    data_Y.append("BrainShift-deLorenzo2012")
    data_C.append(c)
    data_A.append(Common.NH)
    data_M.append(Common.LIN)

    # [Bucki2012]
    model = Linear.Generic("Linear brain-shift compensation for ablation [Bucki2012]", 0, 0.694 * kPa, 0.4)
    model.Compute()
    l, W, dW, H, sigma, P, offset = model.Evaluate(MechanicalTests.Uniaxial, [0.9,1.1])
    data_l = np.concatenate((data_l, l.reshape(1, s)), axis=0)
    data_W = np.concatenate((data_W, W.reshape(1, s)), axis=0)
    data_dW = np.concatenate((data_dW, dW[0].reshape(1, s)), axis=0)
    data_H = np.concatenate((data_H, H[0, 0].reshape(1, s)), axis=0)
    data_sigma = np.concatenate((data_sigma, sigma[0].reshape(1, s)), axis=0)
    data_P = np.concatenate((data_P, P.reshape(1, p)), axis=0)
    data_offset.append(offset)
    data_Y.append("BrainShift-Bucki2012")
    data_C.append(c)
    data_A.append(Common.NH)
    data_M.append(Common.LIN)

    model = Linear.Generic("Linear brain-shift compensation for ablation --ventricles-- [Bucki2012]", 0, 0.01 * kPa, 0.05)
    model.Compute()
    l, W, dW, H, sigma, P, offset = model.Evaluate(MechanicalTests.Uniaxial, [0.9,1.1])
    data_l = np.concatenate((data_l, l.reshape(1, s)), axis=0)
    data_W = np.concatenate((data_W, W.reshape(1, s)), axis=0)
    data_dW = np.concatenate((data_dW, dW[0].reshape(1, s)), axis=0)
    data_H = np.concatenate((data_H, H[0, 0].reshape(1, s)), axis=0)
    data_sigma = np.concatenate((data_sigma, sigma[0].reshape(1, s)), axis=0)
    data_P = np.concatenate((data_P, P.reshape(1, p)), axis=0)
    data_offset.append(offset)
    data_Y.append("BrainShift-ventricles-Bucki2012")
    data_C.append(c)
    data_A.append(Common.AB)
    data_M.append(Common.LIN)

    # [Miller2013]
    model = Linear.Generic("Linear brain-shift compensation for ablation --normal-- [Miller2013]", 0, 3.0 * kPa, 0.45)
    model.Compute()
    l, W, dW, H, sigma, P, offset = model.Evaluate(MechanicalTests.Uniaxial, [0.9,1.1])
    data_l = np.concatenate((data_l, l.reshape(1, s)), axis=0)
    data_W = np.concatenate((data_W, W.reshape(1, s)), axis=0)
    data_dW = np.concatenate((data_dW, dW[0].reshape(1, s)), axis=0)
    data_H = np.concatenate((data_H, H[0, 0].reshape(1, s)), axis=0)
    data_sigma = np.concatenate((data_sigma, sigma[0].reshape(1, s)), axis=0)
    data_Y.append("BrainShift-Miller2013")
    data_C.append(c)
    data_A.append(Common.NH)
    data_M.append(Common.LIN)

    # [Mohammadi2015]
    model = Linear.Generic("Linear brain-shift compensation for ablation [Mohammadi2015]", 0, 0.7 * kPa, 0.42)
    model.Compute()
    l, W, dW, H, sigma, P, offset = model.Evaluate(MechanicalTests.Uniaxial, [0.9,1.1])
    data_l = np.concatenate((data_l, l.reshape(1, s)), axis=0)
    data_W = np.concatenate((data_W, W.reshape(1, s)), axis=0)
    data_dW = np.concatenate((data_dW, dW[0].reshape(1, s)), axis=0)
    data_H = np.concatenate((data_H, H[0, 0].reshape(1, s)), axis=0)
    data_sigma = np.concatenate((data_sigma, sigma[0].reshape(1, s)), axis=0)
    data_P = np.concatenate((data_P, P.reshape(1, p)), axis=0)
    data_offset.append(offset)
    data_Y.append("BrainShift-Mohammadi2015")
    data_C.append(c)
    data_A.append(Common.NH)
    data_M.append(Common.LIN)

    model = Linear.Generic("Linear brain-shift compensation for ablation --ventricles-- [Mohammadi2015]", 0, 0.015 * kPa, 0.05)
    model.Compute()
    l, W, dW, H, sigma, P, offset = model.Evaluate(MechanicalTests.Uniaxial, [0.9,1.1])
    data_l = np.concatenate((data_l, l.reshape(1, s)), axis=0)
    data_W = np.concatenate((data_W, W.reshape(1, s)), axis=0)
    data_dW = np.concatenate((data_dW, dW[0].reshape(1, s)), axis=0)
    data_H = np.concatenate((data_H, H[0, 0].reshape(1, s)), axis=0)
    data_sigma = np.concatenate((data_sigma, sigma[0].reshape(1, s)), axis=0)
    data_P = np.concatenate((data_P, P.reshape(1, p)), axis=0)
    data_offset.append(offset)
    data_Y.append("BrainShift-ventricles-Mohammadi2015")
    data_C.append(c)
    data_A.append(Common.AB)
    data_M.append(Common.LIN)

    # [Morin2016]
    model = Linear.Generic("Linear brain-shift compensation for ablation --normal-- [Morin2016]", 0, 1.5 * kPa, 0.45)
    model.Compute()
    l, W, dW, H, sigma, P, offset = model.Evaluate(MechanicalTests.Uniaxial, [0.9,1.1])
    data_l = np.concatenate((data_l, l.reshape(1, s)), axis=0)
    data_W = np.concatenate((data_W, W.reshape(1, s)), axis=0)
    data_dW = np.concatenate((data_dW, dW[0].reshape(1, s)), axis=0)
    data_H = np.concatenate((data_H, H[0, 0].reshape(1, s)), axis=0)
    data_sigma = np.concatenate((data_sigma, sigma[0].reshape(1, s)), axis=0)
    data_P = np.concatenate((data_P, P.reshape(1, p)), axis=0)
    data_offset.append(offset)
    data_Y.append("BrainShift-Morin2016")
    data_C.append(c)
    data_A.append(Common.NH)
    data_M.append(Common.LIN)

    # [Clatz2003]
    model = Linear.Generic("Linear brain-shift compensation for Parkinson [Clatz2003]", 0, 2.0 * kPa, 0.45)
    model.Compute()
    l, W, dW, H, sigma, P, offset = model.Evaluate(MechanicalTests.Uniaxial, [0.9,1.1])
    data_l = np.concatenate((data_l, l.reshape(1, s)), axis=0)
    data_W = np.concatenate((data_W, W.reshape(1, s)), axis=0)
    data_dW = np.concatenate((data_dW, dW[0].reshape(1, s)), axis=0)
    data_H = np.concatenate((data_H, H[0, 0].reshape(1, s)), axis=0)
    data_sigma = np.concatenate((data_sigma, sigma[0].reshape(1, s)), axis=0)
    data_P = np.concatenate((data_P, P.reshape(1, p)), axis=0)
    data_offset.append(offset)
    data_Y.append("BrainShift-Clatz2003")
    data_C.append(c)
    data_A.append(Common.NH)
    data_M.append(Common.LIN)

    PlotMaterial.Groups("Brain-shift Compensation", stretches, data_l, data_W, data_dW, data_H, data_sigma, data_Y)

    return data_l, data_W, data_dW, data_H, data_sigma, data_P, data_offset, data_Y, data_C, data_A, data_M