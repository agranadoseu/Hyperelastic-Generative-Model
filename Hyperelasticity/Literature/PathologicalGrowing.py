import numpy as np

from Hyperelasticity.utils import Common
from Hyperelasticity.utils import PlotMaterial
from Hyperelasticity.utils import MechanicalTests
from Hyperelasticity import Linear, Nonlinear

kPa = Common.kPa
p = Common.nP
s = Common.sample
c = 2

"""
.........................................................
LINEAR PATHOLOGICAL GROWING REGION
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

    # [Yousefi2013]
    # E +- 0.25
    # v +- 0.145
    model = Linear.Generic("Linear pathological region growing --brain-- [Yousefi2013]", 0, 3.0 * kPa, 0.45)
    model.Compute()
    l, W, dW, H, sigma, P, offset = model.Evaluate(MechanicalTests.Uniaxial, [0.9,1.1])
    data_l[0, :] = l
    data_W[0, :] = W
    data_dW[0, :] = dW[0]
    data_H[0, :] = H[0,0]
    data_sigma[0, :] = sigma[0]
    data_P[0, :] = P
    data_offset.append(offset)
    data_Y[0] = "Pathos-Yousefi2013"
    data_C.append(c)
    data_A.append(Common.NH)
    data_M.append(Common.LIN)
    stretches = model.stretches

    # [Prastawa2009]
    model = Linear.Generic("Linear pathological region growing --brain-- [Prastawa2009]", 0, 0.694 * kPa, 0.4)
    model.Compute()
    l, W, dW, H, sigma, P, offset = model.Evaluate(MechanicalTests.Uniaxial, [0.9,1.1])
    data_l = np.concatenate((data_l, l.reshape(1, s)), axis=0)
    data_W = np.concatenate((data_W, W.reshape(1, s)), axis=0)
    data_dW = np.concatenate((data_dW, dW[0].reshape(1, s)), axis=0)
    data_H = np.concatenate((data_H, H[0, 0].reshape(1, s)), axis=0)
    data_sigma = np.concatenate((data_sigma, sigma[0].reshape(1, s)), axis=0)
    data_P = np.concatenate((data_P, P.reshape(1, p)), axis=0)
    data_offset.append(offset)
    data_Y.append("Pathos-Prastawa2009")
    data_C.append(c)
    data_A.append(Common.NH)
    data_M.append(Common.LIN)

    # model = Linear.Generic("Linear pathological region growing --falx-- [Prastawa2009]", 0, 200.0 * kPa, 0.4)
    # model.Compute()
    # l, W, dW, H, sigma, P, offset = model.Evaluate(MechanicalTests.Uniaxial, [0.9,1.1])
    # data_l = np.concatenate((data_l, l.reshape(1, s)), axis=0)
    # data_W = np.concatenate((data_W, W.reshape(1, s)), axis=0)
    # data_dW = np.concatenate((data_dW, dW[0].reshape(1, s)), axis=0)
    # data_H = np.concatenate((data_H, H[0, 0].reshape(1, s)), axis=0)
    # data_sigma = np.concatenate((data_sigma, sigma[0].reshape(1, s)), axis=0)
    # data_P = np.concatenate((data_P, P.reshape(1, p)), axis=0)
    # data_offset.append(offset)
    # data_Y.append("Pathos-falx-Prastawa2009")
    # data_C.append(c)
    # data_A.append(Common.AB)
    # data_M.append(Common.LIN)

    # [Kyriacou1999]
    # TODO MICCAI exclusion
    # model = Nonlinear.Mihai2015a("Neo-Hookean --WM-- [Kyriacou1999]", 3.0 * kPa)
    # model.Compute()
    # l, W, dW, H, sigma, P, offset = model.Evaluate(MechanicalTests.Uniaxial, [0.8,1.2])
    # data_l = np.concatenate((data_l, l.reshape(1, s)), axis=0)
    # data_W = np.concatenate((data_W, W.reshape(1, s)), axis=0)
    # data_dW = np.concatenate((data_dW, dW[0].reshape(1, s)), axis=0)
    # data_H = np.concatenate((data_H, H[0, 0].reshape(1, s)), axis=0)
    # data_sigma = np.concatenate((data_sigma, sigma[0].reshape(1, s)), axis=0)
    # data_P = np.concatenate((data_P, P.reshape(1, p)), axis=0)
    # data_offset.append(offset)
    # data_Y.append("Pathos-WM-Kyriacou1999")
    # data_C.append(c)
    # data_A.append(Common.WM)
    # data_M.append(Common.HYP)

    # TODO MICCAI exclusion
    # model = Nonlinear.Mihai2015a("Neo-Hookean --GM-- [Kyriacou1999]", 30.0 * kPa)
    # model.Compute()
    # l, W, dW, H, sigma, P, offset = model.Evaluate(MechanicalTests.Uniaxial, [0.8,1.2])
    # data_l = np.concatenate((data_l, l.reshape(1, s)), axis=0)
    # data_W = np.concatenate((data_W, W.reshape(1, s)), axis=0)
    # data_dW = np.concatenate((data_dW, dW[0].reshape(1, s)), axis=0)
    # data_H = np.concatenate((data_H, H[0, 0].reshape(1, s)), axis=0)
    # data_sigma = np.concatenate((data_sigma, sigma[0].reshape(1, s)), axis=0)
    # data_P = np.concatenate((data_P, P.reshape(1, p)), axis=0)
    # data_offset.append(offset)
    # data_Y.append("Pathos-GM-Kyriacou1999")
    # data_C.append(c)
    # data_A.append(Common.GM)
    # data_M.append(Common.HYP)

    # [6]
    model = Nonlinear.Mihai2015a("Neo-Hookean --Tumour-- [Kyriacou1999]", 30.0 * kPa)
    model.Compute()
    l, W, dW, H, sigma, P, offset = model.Evaluate(MechanicalTests.Uniaxial, [0.8,1.2])
    data_l = np.concatenate((data_l, l.reshape(1, s)), axis=0)
    data_W = np.concatenate((data_W, W.reshape(1, s)), axis=0)
    data_dW = np.concatenate((data_dW, dW[0].reshape(1, s)), axis=0)
    data_H = np.concatenate((data_H, H[0, 0].reshape(1, s)), axis=0)
    data_sigma = np.concatenate((data_sigma, sigma[0].reshape(1, s)), axis=0)
    data_P = np.concatenate((data_P, P.reshape(1, p)), axis=0)
    data_offset.append(offset)
    data_Y.append("Pathos-TUMOUR-Kyriacou1999")
    data_C.append(c)
    data_A.append(Common.AB)
    data_M.append(Common.HYP)

    # [Takizawa1994]
    # model = Linear.Generic("Linear pathological region growing --CSF-- [Takizawa1994]", 0, 1.0 * kPa, 0.47)
    # model.Compute()
    # l, W, dW, H, sigma, P, offset = model.Evaluate(MechanicalTests.Uniaxial, [0.9,1.1])
    # data_l = np.concatenate((data_l, l.reshape(1, s)), axis=0)
    # data_W = np.concatenate((data_W, W.reshape(1, s)), axis=0)
    # data_dW = np.concatenate((data_dW, dW[0].reshape(1, s)), axis=0)
    # data_H = np.concatenate((data_H, H[0, 0].reshape(1, s)), axis=0)
    # data_sigma = np.concatenate((data_sigma, sigma[0].reshape(1, s)), axis=0)
    # data_P = np.concatenate((data_P, P.reshape(1, p)), axis=0)
    # data_offset.append(offset)
    # data_Y.append("Pathos-CSF-Takizawa1994")
    # data_C.append(c)
    # data_A.append(Common.AB)
    # data_M.append(Common.LIN)

    # model = Linear.Generic("Linear pathological region growing --falx-- [Takizawa1994]", 0, 100.0 * kPa, 0.47)
    # model.Compute()
    # l, W, dW, H, sigma, P, offset = model.Evaluate(MechanicalTests.Uniaxial, [0.9,1.1])
    # data_l = np.concatenate((data_l, l.reshape(1, s)), axis=0)
    # data_W = np.concatenate((data_W, W.reshape(1, s)), axis=0)
    # data_dW = np.concatenate((data_dW, dW[0].reshape(1, s)), axis=0)
    # data_H = np.concatenate((data_H, H[0, 0].reshape(1, s)), axis=0)
    # data_sigma = np.concatenate((data_sigma, sigma[0].reshape(1, s)), axis=0)
    # data_P = np.concatenate((data_P, P.reshape(1, p)), axis=0)
    # data_offset.append(offset)
    # data_Y.append("Pathos-falx-Takizawa1994")
    # data_C.append(c)
    # data_A.append(Common.AB)
    # data_M.append(Common.LIN)

    model = Linear.Generic("Linear pathological region growing --WM-- [Takizawa1994]", 0, 4.0 * kPa, 0.47)
    model.Compute()
    l, W, dW, H, sigma, P, offset = model.Evaluate(MechanicalTests.Uniaxial, [0.9,1.1])
    data_l = np.concatenate((data_l, l.reshape(1, s)), axis=0)
    data_W = np.concatenate((data_W, W.reshape(1, s)), axis=0)
    data_dW = np.concatenate((data_dW, dW[0].reshape(1, s)), axis=0)
    data_H = np.concatenate((data_H, H[0, 0].reshape(1, s)), axis=0)
    data_sigma = np.concatenate((data_sigma, sigma[0].reshape(1, s)), axis=0)
    data_P = np.concatenate((data_P, P.reshape(1, p)), axis=0)
    data_offset.append(offset)
    data_Y.append("Pathos-WM-Takizawa1994")
    data_C.append(c)
    data_A.append(Common.WM)
    data_M.append(Common.LIN)

    model = Linear.Generic("Linear pathological region growing --GM-- [Takizawa1994]", 0, 8.0 * kPa, 0.47)
    model.Compute()
    l, W, dW, H, sigma, P, offset = model.Evaluate(MechanicalTests.Uniaxial, [0.9,1.1])
    data_l = np.concatenate((data_l, l.reshape(1, s)), axis=0)
    data_W = np.concatenate((data_W, W.reshape(1, s)), axis=0)
    data_dW = np.concatenate((data_dW, dW[0].reshape(1, s)), axis=0)
    data_H = np.concatenate((data_H, H[0,0].reshape(1, s)), axis=0)
    data_sigma = np.concatenate((data_sigma, sigma[0].reshape(1, s)), axis=0)
    data_P = np.concatenate((data_P, P.reshape(1, p)), axis=0)
    data_offset.append(offset)
    data_Y.append("Pathos-GM-Takizawa1994")
    data_C.append(c)
    data_A.append(Common.GM)
    data_M.append(Common.LIN)

    # [Dumpuri2006,Chen2011]
    model = Linear.Generic("Linear pathological region growing --tumour-- [Dumpuri2006,Chen2011]", 0, 100.0 * kPa, 0.45)
    model.Compute()
    l, W, dW, H, sigma, P, offset = model.Evaluate(MechanicalTests.Uniaxial, [0.9,1.1])
    data_l = np.concatenate((data_l, l.reshape(1, s)), axis=0)
    data_W = np.concatenate((data_W, W.reshape(1, s)), axis=0)
    data_dW = np.concatenate((data_dW, dW[0].reshape(1, s)), axis=0)
    data_H = np.concatenate((data_H, H[0, 0].reshape(1, s)), axis=0)
    data_sigma = np.concatenate((data_sigma, sigma[0].reshape(1, s)), axis=0)
    data_P = np.concatenate((data_P, P.reshape(1, p)), axis=0)
    data_offset.append(offset)
    data_Y.append("Pathos-Tumour-Dumpuri2006")
    data_C.append(c)
    data_A.append(Common.AB)
    data_M.append(Common.LIN)

    model = Linear.Generic("Linear brain-shift compensation for ablation --tumour-- [Miller2013]", 0, 9.0 * kPa, 0.45)
    model.Compute()
    l, W, dW, H, sigma, P, offset = model.Evaluate(MechanicalTests.Uniaxial, [0.9, 1.1])
    data_l = np.concatenate((data_l, l.reshape(1, s)), axis=0)
    data_W = np.concatenate((data_W, W.reshape(1, s)), axis=0)
    data_dW = np.concatenate((data_dW, dW[0].reshape(1, s)), axis=0)
    data_H = np.concatenate((data_H, H[0, 0].reshape(1, s)), axis=0)
    data_sigma = np.concatenate((data_sigma, sigma[0].reshape(1, s)), axis=0)
    data_P = np.concatenate((data_P, P.reshape(1, p)), axis=0)
    data_offset.append(offset)
    data_Y.append("BrainShift-tumour-Miller2013")
    data_C.append(c)
    data_A.append(Common.AB)
    data_M.append(Common.LIN)

    model = Linear.Generic("Linear brain-shift compensation for ablation --tumour-- [Morin2016]", 0, 10.0 * kPa, 0.45)
    model.Compute()
    l, W, dW, H, sigma, P, offset = model.Evaluate(MechanicalTests.Uniaxial, [0.9, 1.1])
    data_l = np.concatenate((data_l, l.reshape(1, s)), axis=0)
    data_W = np.concatenate((data_W, W.reshape(1, s)), axis=0)
    data_dW = np.concatenate((data_dW, dW[0].reshape(1, s)), axis=0)
    data_H = np.concatenate((data_H, H[0, 0].reshape(1, s)), axis=0)
    data_sigma = np.concatenate((data_sigma, sigma[0].reshape(1, s)), axis=0)
    data_P = np.concatenate((data_P, P.reshape(1, p)), axis=0)
    data_offset.append(offset)
    data_Y.append("BrainShift-tumour-Morin2016")
    data_C.append(c)
    data_A.append(Common.AB)
    data_M.append(Common.LIN)

    PlotMaterial.Groups("Pathological Region Growing", stretches, data_l, data_W, data_dW, data_H, data_sigma, data_Y)

    return data_l, data_W, data_dW, data_H, data_sigma, data_P, data_offset, data_Y, data_C, data_A, data_M