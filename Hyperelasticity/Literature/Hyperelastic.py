import numpy as np

from Hyperelasticity.utils import Common
from Hyperelasticity.utils import PlotMaterial
from Hyperelasticity.utils import MechanicalTests
from Hyperelasticity.utils import Postulates
from Hyperelasticity import Linear, Nonlinear

kPa = Common.kPa
p = Common.nP
s = Common.sample
c = 6


"""
.........................................................
NON-LINEAR
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

    ''' [Mihai2017] [38] '''
    # TODO check - too flat and fitting is bad
    # TODO Recent papers to add
    model = Nonlinear.Mihai2017a("Ogden 1-term [Mihai2017]", -0.0939 * kPa, -4.0250)
    model.Compute()
    l, W, dW, H, sigma, P, offset = model.Evaluate(MechanicalTests.Uniaxial, [0.8,1.2])
    data_l[0, :] = l
    data_W[0, :] = W
    data_dW[0, :] = dW[0]
    data_H[0, :] = H[0,0]
    data_sigma[0, :] = sigma[0]
    data_P[0, :] = P
    data_offset.append(offset)
    data_Y[0] = "Hyperelastic-Ogden1-Mihai2017"
    data_C.append(c)
    data_A.append(Common.NH)
    data_M.append(Common.HYP)
    stretches = model.stretches

    model = Nonlinear.Mihai2017b("3-term Mooney-Rivlin: Ogden 1-term + MR 2-term [Mihai2017]", 0.0653 * kPa, 7.1813, -3.8201 * kPa, 3.5376 * kPa)
    model.Compute()
    l, W, dW, H, sigma, P, offset = model.Evaluate(MechanicalTests.Uniaxial, [0.8,1.2])
    data_l = np.concatenate((data_l, l.reshape(1, s)), axis=0)
    data_W = np.concatenate((data_W, W.reshape(1, s)), axis=0)
    data_dW = np.concatenate((data_dW, dW[0].reshape(1, s)), axis=0)
    data_H = np.concatenate((data_H, H[0,0].reshape(1, s)), axis=0)
    data_sigma = np.concatenate((data_sigma, sigma[0].reshape(1, s)), axis=0)
    data_P = np.concatenate((data_P, P.reshape(1, p)), axis=0)
    data_offset.append(offset)
    data_Y.append("Hyperelastic-MR3-Mihai2017")
    data_C.append(c)
    data_A.append(Common.NH)
    data_M.append(Common.HYP)

    # TODO check - negative
    # model = Nonlinear.Mihai2017c("3-term Ogden [Mihai2017]", 3, [-5.5090 * kPa, 2.9269 * kPa, 1.4653 * kPa], [2, -2, 4])
    # model.Compute()
    # l, W, dW, H, sigma, P, offset = model.Evaluate(MechanicalTests.Uniaxial, [0.8,1.2])
    # data_l = np.concatenate((data_l, l.reshape(1, s)), axis=0)
    # data_W = np.concatenate((data_W, W.reshape(1, s)), axis=0)
    # data_dW = np.concatenate((data_dW, dW[0].reshape(1, s)), axis=0)
    # data_H = np.concatenate((data_H, H[0, 0].reshape(1, s)), axis=0)
    # data_sigma = np.concatenate((data_sigma, sigma[0].reshape(1, s)), axis=0)
    # data_P = np.concatenate((data_P, P.reshape(1, p)), axis=0)
    # data_offset.append(offset)
    # data_Y.append("Hyperelastic-Ogden3-Mihai2017")
    # data_C.append(c)
    # data_A.append(Common.NH)
    # data_M.append(Common.HYP)

    ''' [Mihai2015] '''
    # model = Nonlinear.Mihai2015a("Neo-Hookean [Mihai2015]", 333.28 * kPa)
    model = Nonlinear.Mihai2015a("Neo-Hookean [Mihai2015]", 0.33328 * kPa)
    model.Compute()
    l, W, dW, H, sigma, P, offset = model.Evaluate(MechanicalTests.Uniaxial, [0.6,1.4])
    data_l = np.concatenate((data_l, l.reshape(1, s)), axis=0)
    data_W = np.concatenate((data_W, W.reshape(1, s)), axis=0)
    data_dW = np.concatenate((data_dW, dW[0].reshape(1, s)), axis=0)
    data_H = np.concatenate((data_H, H[0, 0].reshape(1, s)), axis=0)
    data_sigma = np.concatenate((data_sigma, sigma[0].reshape(1, s)), axis=0)
    data_P = np.concatenate((data_P, P.reshape(1, p)), axis=0)
    data_offset.append(offset)
    data_Y.append("Hyperelastic-NH-Mihai2015")
    data_C.append(c)
    data_A.append(Common.NH)
    data_M.append(Common.HYP)

    # TODO check - prediction is poor
    # model = Nonlinear.Mihai2015b("Mooney-Rivlin [Mihai2015]", 0.28 * kPa, 333.0 * kPa)
    model = Nonlinear.Mihai2015b("Mooney-Rivlin [Mihai2015]", 0.28 * kPa, 0.333 * kPa)
    model.Compute()
    l, W, dW, H, sigma, P, offset = model.Evaluate(MechanicalTests.Uniaxial, [0.6,1.4])
    data_l = np.concatenate((data_l, l.reshape(1, s)), axis=0)
    data_W = np.concatenate((data_W, W.reshape(1, s)), axis=0)
    data_dW = np.concatenate((data_dW, dW[0].reshape(1, s)), axis=0)
    data_H = np.concatenate((data_H, H[0, 0].reshape(1, s)), axis=0)
    data_sigma = np.concatenate((data_sigma, sigma[0].reshape(1, s)), axis=0)
    data_P = np.concatenate((data_P, P.reshape(1, p)), axis=0)
    data_offset.append(offset)
    data_Y.append("Hyperelastic-MR-Mihai2015")
    data_C.append(c)
    data_A.append(Common.NH)
    data_M.append(Common.HYP)

    # TODO check - fit crashes
    # model = Nonlinear.Mihai2015c("Fung [Mihai2015]", 166.64 * kPa, 2.4974)
    model = Nonlinear.Mihai2015c("Fung [Mihai2015]", 0.16664 * kPa, 2.4974)
    model.Compute()
    l, W, dW, H, sigma, P, offset = model.Evaluate(MechanicalTests.Uniaxial, [0.6,1.4])
    data_l = np.concatenate((data_l, l.reshape(1, s)), axis=0)
    data_W = np.concatenate((data_W, W.reshape(1, s)), axis=0)
    data_dW = np.concatenate((data_dW, dW[0].reshape(1, s)), axis=0)
    data_H = np.concatenate((data_H, H[0, 0].reshape(1, s)), axis=0)
    data_sigma = np.concatenate((data_sigma, sigma[0].reshape(1, s)), axis=0)
    data_P = np.concatenate((data_P, P.reshape(1, p)), axis=0)
    data_offset.append(offset)
    data_Y.append("Hyperelastic-Fung-Mihai2015")
    data_C.append(c)
    data_A.append(Common.NH)
    data_M.append(Common.HYP)

    # TODO check fit nan
    # model = Nonlinear.Mihai2015d("Gent [Mihai2015]", 333.28 * kPa, 0.9918)
    model = Nonlinear.Mihai2015d("Gent [Mihai2015]", 0.33328 * kPa, 0.9918)
    model.Compute()
    l, W, dW, H, sigma, P, offset = model.Evaluate(MechanicalTests.Uniaxial, [0.6,1.4])
    data_l = np.concatenate((data_l, l.reshape(1, s)), axis=0)
    data_W = np.concatenate((data_W, W.reshape(1, s)), axis=0)
    data_dW = np.concatenate((data_dW, dW[0].reshape(1, s)), axis=0)
    data_H = np.concatenate((data_H, H[0, 0].reshape(1, s)), axis=0)
    data_sigma = np.concatenate((data_sigma, sigma[0].reshape(1, s)), axis=0)
    data_P = np.concatenate((data_P, P.reshape(1, p)), axis=0)
    data_offset.append(offset)
    data_Y.append("Hyperelastic-Gent-Mihai2015")
    data_C.append(c)
    data_A.append(Common.NH)
    data_M.append(Common.HYP)

    # TODO check - change optimisation params
    # model = Nonlinear.Mihai2015e("Ogden 3-term [Mihai2015]", 3, [-3543 * kPa, -2723 * kPa, 654 * kPa], [1, -1, 2])
    model = Nonlinear.Mihai2015e("Ogden 3-term [Mihai2015]", 3, [-3.543 * kPa, -2.723 * kPa, 0.654 * kPa], [1, -1, 2])
    model.Compute()
    l, W, dW, H, sigma, P, offset = model.Evaluate(MechanicalTests.Uniaxial, [0.6,1.4])
    # l, W, dW, H, sigma, P, offset = model.Evaluate(MechanicalTests.Uniaxial, [0.01, 5])
    data_l = np.concatenate((data_l, l.reshape(1, s)), axis=0)
    data_W = np.concatenate((data_W, W.reshape(1, s)), axis=0)
    data_dW = np.concatenate((data_dW, dW[0].reshape(1, s)), axis=0)
    data_H = np.concatenate((data_H, H[0, 0].reshape(1, s)), axis=0)
    data_sigma = np.concatenate((data_sigma, sigma[0].reshape(1, s)), axis=0)
    data_P = np.concatenate((data_P, P.reshape(1, p)), axis=0)
    data_offset.append(offset)
    data_Y.append("Hyperelastic-Ogden3-Mihai2015")
    data_C.append(c)
    data_A.append(Common.NH)
    data_M.append(Common.HYP)

    # TODO check - strain curve starts from megative values (consider only narrower window)
    # model = Nonlinear.Mihai2015e("Ogden 4-term [Mihai2015]", 4, [-5877 * kPa, -5043 * kPa, 1161 * kPa, 501 * kPa], [1, -1, 2, -2])
    model = Nonlinear.Mihai2015e("Ogden 4-term [Mihai2015]", 4, [-5.877 * kPa, -5.043 * kPa, 1.161 * kPa, 0.501 * kPa], [1, -1, 2, -2])
    model.Compute()
    l, W, dW, H, sigma, P, offset = model.Evaluate(MechanicalTests.Uniaxial, [0.6,1.4])
    data_l = np.concatenate((data_l, l.reshape(1, s)), axis=0)
    data_W = np.concatenate((data_W, W.reshape(1, s)), axis=0)
    data_dW = np.concatenate((data_dW, dW[0].reshape(1, s)), axis=0)
    data_H = np.concatenate((data_H, H[0, 0].reshape(1, s)), axis=0)
    data_sigma = np.concatenate((data_sigma, sigma[0].reshape(1, s)), axis=0)
    data_P = np.concatenate((data_P, P.reshape(1, p)), axis=0)
    data_offset.append(offset)
    data_Y.append("Hyperelastic-Ogden4-Mihai2015")
    data_C.append(c)
    data_A.append(Common.NH)
    data_M.append(Common.HYP)

    # TODO check - consider only narrower window
    # model = Nonlinear.Mihai2015e("Ogden 5-term [Mihai2015]", 5, [-34399 * kPa, -18718 * kPa, 14509 * kPa, 2947 * kPa, -2349 * kPa], [1, -1, 2, -2, 3])
    model = Nonlinear.Mihai2015e("Ogden 5-term [Mihai2015]", 5, [-34.399 * kPa, -18.718 * kPa, 14.509 * kPa, 2.947 * kPa, -2.349 * kPa], [1, -1, 2, -2, 3])
    model.Compute()
    l, W, dW, H, sigma, P, offset = model.Evaluate(MechanicalTests.Uniaxial, [0.6,1.4])
    data_l = np.concatenate((data_l, l.reshape(1, s)), axis=0)
    data_W = np.concatenate((data_W, W.reshape(1, s)), axis=0)
    data_dW = np.concatenate((data_dW, dW[0].reshape(1, s)), axis=0)
    data_H = np.concatenate((data_H, H[0, 0].reshape(1, s)), axis=0)
    data_sigma = np.concatenate((data_sigma, sigma[0].reshape(1, s)), axis=0)
    data_P = np.concatenate((data_P, P.reshape(1, p)), axis=0)
    data_offset.append(offset)
    data_Y.append("Hyperelastic-Ogden5-Mihai2015")
    data_C.append(c)
    data_A.append(Common.NH)
    data_M.append(Common.HYP)

    # model = Nonlinear.Mihai2015e("Ogden 6-term [Mihai2015]", 6, [1189 * kPa, 16855 * kPa, 1444 * kPa, -10108 * kPa, -458 * kPa, 1889 * kPa], [1, -1, 2, -2, 3, -3])
    model = Nonlinear.Mihai2015e("Ogden 6-term [Mihai2015]", 6, [1.189 * kPa, 16.855 * kPa, 1.444 * kPa, -10.108 * kPa, -0.458 * kPa, 1.889 * kPa], [1, -1, 2, -2, 3, -3])
    model.Compute()
    l, W, dW, H, sigma, P, offset = model.Evaluate(MechanicalTests.Uniaxial, [0.6,1.4])
    data_l = np.concatenate((data_l, l.reshape(1, s)), axis=0)
    data_W = np.concatenate((data_W, W.reshape(1, s)), axis=0)
    data_dW = np.concatenate((data_dW, dW[0].reshape(1, s)), axis=0)
    data_H = np.concatenate((data_H, H[0, 0].reshape(1, s)), axis=0)
    data_sigma = np.concatenate((data_sigma, sigma[0].reshape(1, s)), axis=0)
    data_P = np.concatenate((data_P, P.reshape(1, p)), axis=0)
    data_offset.append(offset)
    data_Y.append("Hyperelastic-Ogden6-Mihai2015")
    data_C.append(c)
    data_A.append(Common.NH)
    data_M.append(Common.HYP)

    # model = Nonlinear.Mihai2015e("Ogden 7-term [Mihai2015]", 7, [-187150 * kPa, -91970 * kPa, 109290 * kPa, 23200 * kPa, -33290 * kPa, -2290 * kPa, 4100 * kPa], [1, -1, 2, -2, 3, -3, 4])
    model = Nonlinear.Mihai2015e("Ogden 7-term [Mihai2015]", 7, [-187.150 * kPa, -91.970 * kPa, 109.290 * kPa, 23.200 * kPa, -33.290 * kPa, -2.290 * kPa, 4.100 * kPa], [1, -1, 2, -2, 3, -3, 4])
    model.Compute()
    l, W, dW, H, sigma, P, offset = model.Evaluate(MechanicalTests.Uniaxial, [0.6,1.4])
    data_l = np.concatenate((data_l, l.reshape(1, s)), axis=0)
    data_W = np.concatenate((data_W, W.reshape(1, s)), axis=0)
    data_dW = np.concatenate((data_dW, dW[0].reshape(1, s)), axis=0)
    data_H = np.concatenate((data_H, H[0, 0].reshape(1, s)), axis=0)
    data_sigma = np.concatenate((data_sigma, sigma[0].reshape(1, s)), axis=0)
    data_P = np.concatenate((data_P, P.reshape(1, p)), axis=0)
    data_offset.append(offset)
    data_Y.append("Hyperelastic-Ogden7-Mihai2015")
    data_C.append(c)
    data_A.append(Common.NH)
    data_M.append(Common.HYP)

    # TODO check - consider only window
    # model = Nonlinear.Mihai2015e("Ogden 8-term [Mihai2015]", 8, [-639530 * kPa, -544840 * kPa, 322660 * kPa, 237040 * kPa, -88640 * kPa, -57830 * kPa, 10150 * kPa, 6080 * kPa], [1, -1, 2, -2, 3, -3, 4, -4])
    model = Nonlinear.Mihai2015e("Ogden 8-term [Mihai2015]", 8, [-639.530 * kPa, -544.840 * kPa, 322.660 * kPa, 237.040 * kPa, -88.640 * kPa, -57.830 * kPa, 10.150 * kPa, 6.080 * kPa], [1, -1, 2, -2, 3, -3, 4, -4])
    model.Compute()
    l, W, dW, H, sigma, P, offset = model.Evaluate(MechanicalTests.Uniaxial, [0.6,1.4])
    data_l = np.concatenate((data_l, l.reshape(1, s)), axis=0)
    data_W = np.concatenate((data_W, W.reshape(1, s)), axis=0)
    data_dW = np.concatenate((data_dW, dW[0].reshape(1, s)), axis=0)
    data_H = np.concatenate((data_H, H[0, 0].reshape(1, s)), axis=0)
    data_sigma = np.concatenate((data_sigma, sigma[0].reshape(1, s)), axis=0)
    data_P = np.concatenate((data_P, P.reshape(1, p)), axis=0)
    data_offset.append(offset)
    data_Y.append("Hyperelastic-Ogden8-Mihai2015")
    data_C.append(c)
    data_A.append(Common.NH)
    data_M.append(Common.HYP)

    ''' [Laksari2012] from [Morin2017] '''
    # TODO c10=[+-0.06], c01=[+-0.05], c11=[+-0.01], K=[+-0.26]
    # model = Nonlinear.Laksari2012("Mooney Rivlin 3-term + Ogden 2-param [Laksari2012]", -1.34 * kPa, 1.83 * kPa, 0.29 * kPa, 46, 100, 0.49)   # compressible
    model = Nonlinear.Laksari2012("Mooney Rivlin 3-term + Ogden 2-param [Laksari2012]", -1.01 * kPa, 1.49 * kPa, 0.19 * kPa, 2.38, 100, 0.49)   # incompressible
    model.Compute()
    l, W, dW, H, sigma, P, offset = model.Evaluate(MechanicalTests.Uniaxial, [0.7,1.3])
    data_l = np.concatenate((data_l, l.reshape(1, s)), axis=0)
    data_W = np.concatenate((data_W, W.reshape(1, s)), axis=0)
    data_dW = np.concatenate((data_dW, dW[0].reshape(1, s)), axis=0)
    data_H = np.concatenate((data_H, H[0, 0].reshape(1, s)), axis=0)
    data_sigma = np.concatenate((data_sigma, sigma[0].reshape(1, s)), axis=0)
    data_P = np.concatenate((data_P, P.reshape(1, p)), axis=0)
    data_offset.append(offset)
    data_Y.append("Hyperelastic-MR3-Laksari2012")
    data_C.append(c)
    data_A.append(Common.NH)
    data_M.append(Common.HYP)

    ''' [Schiavone2009] from [Morin2017] '''
    # TODO MICCAI assumption for tension
    model = Nonlinear.Schiavone2009("Mooney Rivlin 2-term [Schiavone2009]", 0.24 * kPa, 3.42 * kPa)
    model.Compute()
    # l, W, dW, H, sigma, P, offset = model.Evaluate(MechanicalTests.Uniaxial, [1.0,1.45])
    l, W, dW, H, sigma, P, offset = model.Evaluate(MechanicalTests.Uniaxial, [0.55, 1.45])
    data_l = np.concatenate((data_l, l.reshape(1, s)), axis=0)
    data_W = np.concatenate((data_W, W.reshape(1, s)), axis=0)
    data_dW = np.concatenate((data_dW, dW[0].reshape(1, s)), axis=0)
    data_H = np.concatenate((data_H, H[0, 0].reshape(1, s)), axis=0)
    data_sigma = np.concatenate((data_sigma, sigma[0].reshape(1, s)), axis=0)
    data_P = np.concatenate((data_P, P.reshape(1, p)), axis=0)
    data_offset.append(offset)
    data_Y.append("Hyperelastic-MR2-Schiavone2009")
    data_C.append(c)
    data_A.append(Common.NH)
    data_M.append(Common.HYP)

    ''' [Miller2002] '''
    model = Nonlinear.Miller2002("Ogden mod 1-term [Miller2002]", 0.842 * kPa, -4.7)
    model.Compute()
    l, W, dW, H, sigma, P, offset = model.Evaluate(MechanicalTests.Uniaxial, [0.7, 1.3])
    data_l = np.concatenate((data_l, l.reshape(1, s)), axis=0)
    data_W = np.concatenate((data_W, W.reshape(1, s)), axis=0)
    data_dW = np.concatenate((data_dW, dW[0].reshape(1, s)), axis=0)
    data_H = np.concatenate((data_H, H[0, 0].reshape(1, s)), axis=0)
    data_sigma = np.concatenate((data_sigma, sigma[0].reshape(1, s)), axis=0)
    data_P = np.concatenate((data_P, P.reshape(1, p)), axis=0)
    data_offset.append(offset)
    data_Y.append("Hyperelastic-OgdenMod1-Miller2002")
    data_C.append(c)
    data_A.append(Common.NH)
    data_M.append(Common.HYP)

    # model = Nonlinear.Mihai2015e("Ogden 3-term Rubber", 3, [6300 * kPa, 1.2 * kPa, -10 * kPa], [1.3/2.0, 5.0/2.0, -2.0/2.0])
    # model.Compute()
    # # l, W, dW, H, sigma, P, offset = model.Evaluate(MechanicalTests.Uniaxial, [0.8,1.2])
    # l, W, dW, H, sigma, P, offset = model.Evaluate(MechanicalTests.Uniaxial, [0.001, 5])
    # data_l = np.concatenate((data_l, l.reshape(1, s)), axis=0)
    # data_W = np.concatenate((data_W, W.reshape(1, s)), axis=0)
    # data_dW = np.concatenate((data_dW, dW[0].reshape(1, s)), axis=0)
    # data_H = np.concatenate((data_H, H[0, 0].reshape(1, s)), axis=0)
    # data_sigma = np.concatenate((data_sigma, sigma[0].reshape(1, s)), axis=0)
    # data_P = np.concatenate((data_P, P.reshape(1,p)), axis=0)
    # data_offset.append(offset)
    # data_Y.append("Hyperelastic-Ogden3-Mihai2015")
    # data_C.append(c)
    # data_A.append(Common.NH)
    # data_M.append(Common.HYP)


    ''' ############################################################################################# '''
    ''' Budday NH [53]'''
    model = Nonlinear.Budday2017a("Neo-Hookean CC [Budday2017]", 0.99 * kPa, 0.29 * kPa)
    model.Compute()
    l, W, dW, H, sigma, P, offset = model.Evaluate(MechanicalTests.Uniaxial, [0.9, 1.1])
    data_l = np.concatenate((data_l, l.reshape(1, s)), axis=0)
    data_W = np.concatenate((data_W, W.reshape(1, s)), axis=0)
    data_dW = np.concatenate((data_dW, dW[0].reshape(1, s)), axis=0)
    data_H = np.concatenate((data_H, H[0, 0].reshape(1, s)), axis=0)
    data_sigma = np.concatenate((data_sigma, sigma[0].reshape(1, s)), axis=0)
    data_P = np.concatenate((data_P, P.reshape(1, p)), axis=0)
    data_offset.append(offset)
    data_Y.append("Hyperelastic-NH-CC-Budday2017")
    data_C.append(c)
    data_A.append(Common.WM)
    data_M.append(Common.HYP)

    model = Nonlinear.Budday2017a("Neo-Hookean CR [Budday2017]", 1.75 * kPa, 0.52 * kPa)
    model.Compute()
    l, W, dW, H, sigma, P, offset = model.Evaluate(MechanicalTests.Uniaxial, [0.9, 1.1])
    data_l = np.concatenate((data_l, l.reshape(1, s)), axis=0)
    data_W = np.concatenate((data_W, W.reshape(1, s)), axis=0)
    data_dW = np.concatenate((data_dW, dW[0].reshape(1, s)), axis=0)
    data_H = np.concatenate((data_H, H[0, 0].reshape(1, s)), axis=0)
    data_sigma = np.concatenate((data_sigma, sigma[0].reshape(1, s)), axis=0)
    data_P = np.concatenate((data_P, P.reshape(1, p)), axis=0)
    data_offset.append(offset)
    data_Y.append("Hyperelastic-NH-CR-Budday2017")
    data_C.append(c)
    data_A.append(Common.WM)
    data_M.append(Common.HYP)

    # [55]
    model = Nonlinear.Budday2017a("Neo-Hookean BG [Budday2017]", 1.37 * kPa, 0.57 * kPa)
    model.Compute()
    l, W, dW, H, sigma, P, offset = model.Evaluate(MechanicalTests.Uniaxial, [0.9, 1.1])
    data_l = np.concatenate((data_l, l.reshape(1, s)), axis=0)
    data_W = np.concatenate((data_W, W.reshape(1, s)), axis=0)
    data_dW = np.concatenate((data_dW, dW[0].reshape(1, s)), axis=0)
    data_H = np.concatenate((data_H, H[0, 0].reshape(1, s)), axis=0)
    data_sigma = np.concatenate((data_sigma, sigma[0].reshape(1, s)), axis=0)
    data_P = np.concatenate((data_P, P.reshape(1, p)), axis=0)
    data_offset.append(offset)
    data_Y.append("Hyperelastic-NH-BG-Budday2017")
    data_C.append(c)
    data_A.append(Common.GM)
    data_M.append(Common.HYP)

    model = Nonlinear.Budday2017a("Neo-Hookean C [Budday2017]", 2.80 * kPa, 1.22 * kPa)
    model.Compute()
    l, W, dW, H, sigma, P, offset = model.Evaluate(MechanicalTests.Uniaxial, [0.9, 1.1])
    data_l = np.concatenate((data_l, l.reshape(1, s)), axis=0)
    data_W = np.concatenate((data_W, W.reshape(1, s)), axis=0)
    data_dW = np.concatenate((data_dW, dW[0].reshape(1, s)), axis=0)
    data_H = np.concatenate((data_H, H[0, 0].reshape(1, s)), axis=0)
    data_sigma = np.concatenate((data_sigma, sigma[0].reshape(1, s)), axis=0)
    data_P = np.concatenate((data_P, P.reshape(1, p)), axis=0)
    data_offset.append(offset)
    data_Y.append("Hyperelastic-NH-C-Budday2017")
    data_C.append(c)
    data_A.append(Common.GM)
    data_M.append(Common.HYP)

    ''' Budday MR [57]'''
    model = Nonlinear.Budday2017b("Mooney-Rivlin CC [Budday2017]", 0.92*kPa, 0.0*kPa, 0.29*kPa, 0.13*kPa)
    model.Compute()
    l, W, dW, H, sigma, P, offset = model.Evaluate(MechanicalTests.Uniaxial, [0.9, 1.1])
    data_l = np.concatenate((data_l, l.reshape(1, s)), axis=0)
    data_W = np.concatenate((data_W, W.reshape(1, s)), axis=0)
    data_dW = np.concatenate((data_dW, dW[0].reshape(1, s)), axis=0)
    data_H = np.concatenate((data_H, H[0, 0].reshape(1, s)), axis=0)
    data_sigma = np.concatenate((data_sigma, sigma[0].reshape(1, s)), axis=0)
    data_P = np.concatenate((data_P, P.reshape(1, p)), axis=0)
    data_offset.append(offset)
    data_Y.append("Hyperelastic-MR-CC-Budday2017")
    data_C.append(c)
    data_A.append(Common.WM)
    data_M.append(Common.HYP)

    model = Nonlinear.Budday2017b("Mooney-Rivlin CR [Budday2017]", 1.62 * kPa, 0.0 * kPa, 0.52 * kPa, 0.26 * kPa)
    model.Compute()
    l, W, dW, H, sigma, P, offset = model.Evaluate(MechanicalTests.Uniaxial, [0.9, 1.1])
    data_l = np.concatenate((data_l, l.reshape(1, s)), axis=0)
    data_W = np.concatenate((data_W, W.reshape(1, s)), axis=0)
    data_dW = np.concatenate((data_dW, dW[0].reshape(1, s)), axis=0)
    data_H = np.concatenate((data_H, H[0, 0].reshape(1, s)), axis=0)
    data_sigma = np.concatenate((data_sigma, sigma[0].reshape(1, s)), axis=0)
    data_P = np.concatenate((data_P, P.reshape(1, p)), axis=0)
    data_offset.append(offset)
    data_Y.append("Hyperelastic-MR-CR-Budday2017")
    data_C.append(c)
    data_A.append(Common.WM)
    data_M.append(Common.HYP)

    # [59]
    model = Nonlinear.Budday2017b("Mooney-Rivlin BG [Budday2017]", 1.27 * kPa, 0.0 * kPa, 0.57 * kPa, 0.29 * kPa)
    model.Compute()
    l, W, dW, H, sigma, P, offset = model.Evaluate(MechanicalTests.Uniaxial, [0.9, 1.1])
    data_l = np.concatenate((data_l, l.reshape(1, s)), axis=0)
    data_W = np.concatenate((data_W, W.reshape(1, s)), axis=0)
    data_dW = np.concatenate((data_dW, dW[0].reshape(1, s)), axis=0)
    data_H = np.concatenate((data_H, H[0, 0].reshape(1, s)), axis=0)
    data_sigma = np.concatenate((data_sigma, sigma[0].reshape(1, s)), axis=0)
    data_P = np.concatenate((data_P, P.reshape(1, p)), axis=0)
    data_offset.append(offset)
    data_Y.append("Hyperelastic-MR-BG-Budday2017")
    data_C.append(c)
    data_A.append(Common.GM)
    data_M.append(Common.HYP)

    model = Nonlinear.Budday2017b("Mooney-Rivlin C [Budday2017]", 2.59 * kPa, 0.0 * kPa, 1.22 * kPa, 0.61 * kPa)
    model.Compute()
    l, W, dW, H, sigma, P, offset = model.Evaluate(MechanicalTests.Uniaxial, [0.9, 1.1])
    data_l = np.concatenate((data_l, l.reshape(1, s)), axis=0)
    data_W = np.concatenate((data_W, W.reshape(1, s)), axis=0)
    data_dW = np.concatenate((data_dW, dW[0].reshape(1, s)), axis=0)
    data_H = np.concatenate((data_H, H[0, 0].reshape(1, s)), axis=0)
    data_sigma = np.concatenate((data_sigma, sigma[0].reshape(1, s)), axis=0)
    data_P = np.concatenate((data_P, P.reshape(1, p)), axis=0)
    data_offset.append(offset)
    data_Y.append("Hyperelastic-MR-C-Budday2017")
    data_C.append(c)
    data_A.append(Common.GM)
    data_M.append(Common.HYP)

    ''' Budday DMR [61] '''
    model = Nonlinear.Budday2017c("Demiray CC [Budday2017]", 0.55 * kPa, 28.4, 0.28 * kPa, 1.29)
    model.Compute()
    l, W, dW, H, sigma, P, offset = model.Evaluate(MechanicalTests.Uniaxial, [0.9, 1.1])
    data_l = np.concatenate((data_l, l.reshape(1, s)), axis=0)
    data_W = np.concatenate((data_W, W.reshape(1, s)), axis=0)
    data_dW = np.concatenate((data_dW, dW[0].reshape(1, s)), axis=0)
    data_H = np.concatenate((data_H, H[0, 0].reshape(1, s)), axis=0)
    data_sigma = np.concatenate((data_sigma, sigma[0].reshape(1, s)), axis=0)
    data_P = np.concatenate((data_P, P.reshape(1, p)), axis=0)
    data_offset.append(offset)
    data_Y.append("Hyperelastic-DMR-CC-Budday2017")
    data_C.append(c)
    data_A.append(Common.WM)
    data_M.append(Common.HYP)

    model = Nonlinear.Budday2017c("Demiray CR [Budday2017]", 1.05 * kPa, 24.6, 0.48 * kPa, 4.37)
    model.Compute()
    l, W, dW, H, sigma, P, offset = model.Evaluate(MechanicalTests.Uniaxial, [0.9, 1.1])
    data_l = np.concatenate((data_l, l.reshape(1, s)), axis=0)
    data_W = np.concatenate((data_W, W.reshape(1, s)), axis=0)
    data_dW = np.concatenate((data_dW, dW[0].reshape(1, s)), axis=0)
    data_H = np.concatenate((data_H, H[0, 0].reshape(1, s)), axis=0)
    data_sigma = np.concatenate((data_sigma, sigma[0].reshape(1, s)), axis=0)
    data_P = np.concatenate((data_P, P.reshape(1, p)), axis=0)
    data_offset.append(offset)
    data_Y.append("Hyperelastic-DMR-CR-Budday2017")
    data_C.append(c)
    data_A.append(Common.WM)
    data_M.append(Common.HYP)

    # [63]
    model = Nonlinear.Budday2017c("Demiray BG [Budday2017]", 0.97 * kPa, 17.1, 0.52 * kPa, 6.03)
    model.Compute()
    l, W, dW, H, sigma, P, offset = model.Evaluate(MechanicalTests.Uniaxial, [0.9, 1.1])
    data_l = np.concatenate((data_l, l.reshape(1, s)), axis=0)
    data_W = np.concatenate((data_W, W.reshape(1, s)), axis=0)
    data_dW = np.concatenate((data_dW, dW[0].reshape(1, s)), axis=0)
    data_H = np.concatenate((data_H, H[0, 0].reshape(1, s)), axis=0)
    data_sigma = np.concatenate((data_sigma, sigma[0].reshape(1, s)), axis=0)
    data_P = np.concatenate((data_P, P.reshape(1, p)), axis=0)
    data_offset.append(offset)
    data_Y.append("Hyperelastic-DMR-BG-Budday2017")
    data_C.append(c)
    data_A.append(Common.GM)
    data_M.append(Common.HYP)

    model = Nonlinear.Budday2017c("Demiray C [Budday2017]", 1.90 * kPa, 18.8, 0.91 * kPa, 16.7)
    model.Compute()
    l, W, dW, H, sigma, P, offset = model.Evaluate(MechanicalTests.Uniaxial, [0.9, 1.1])
    data_l = np.concatenate((data_l, l.reshape(1, s)), axis=0)
    data_W = np.concatenate((data_W, W.reshape(1, s)), axis=0)
    data_dW = np.concatenate((data_dW, dW[0].reshape(1, s)), axis=0)
    data_H = np.concatenate((data_H, H[0, 0].reshape(1, s)), axis=0)
    data_sigma = np.concatenate((data_sigma, sigma[0].reshape(1, s)), axis=0)
    data_P = np.concatenate((data_P, P.reshape(1, p)), axis=0)
    data_offset.append(offset)
    data_Y.append("Hyperelastic-DMR-C-Budday2017")
    data_C.append(c)
    data_A.append(Common.GM)
    data_M.append(Common.HYP)

    ''' Budday Gnt [65] '''
    model = Nonlinear.Budday2017d("Gent CC [Budday2017]", 0.62 * kPa, 0.06, 0.29 * kPa, 85.0)
    model.Compute()
    l, W, dW, H, sigma, P, offset = model.Evaluate(MechanicalTests.Uniaxial, [0.9, 1.1])
    data_l = np.concatenate((data_l, l.reshape(1, s)), axis=0)
    data_W = np.concatenate((data_W, W.reshape(1, s)), axis=0)
    data_dW = np.concatenate((data_dW, dW[0].reshape(1, s)), axis=0)
    data_H = np.concatenate((data_H, H[0, 0].reshape(1, s)), axis=0)
    data_sigma = np.concatenate((data_sigma, sigma[0].reshape(1, s)), axis=0)
    data_P = np.concatenate((data_P, P.reshape(1, p)), axis=0)
    data_offset.append(offset)
    data_Y.append("Hyperelastic-Gent-CC-Budday2017")
    data_C.append(c)
    data_A.append(Common.WM)
    data_M.append(Common.HYP)

    model = Nonlinear.Budday2017d("Gent CR [Budday2017]", 1.16 * kPa, 0.06, 0.48 * kPa, 0.23)
    model.Compute()
    l, W, dW, H, sigma, P, offset = model.Evaluate(MechanicalTests.Uniaxial, [0.9, 1.1])
    data_l = np.concatenate((data_l, l.reshape(1, s)), axis=0)
    data_W = np.concatenate((data_W, W.reshape(1, s)), axis=0)
    data_dW = np.concatenate((data_dW, dW[0].reshape(1, s)), axis=0)
    data_H = np.concatenate((data_H, H[0, 0].reshape(1, s)), axis=0)
    data_sigma = np.concatenate((data_sigma, sigma[0].reshape(1, s)), axis=0)
    data_P = np.concatenate((data_P, P.reshape(1, p)), axis=0)
    data_offset.append(offset)
    data_Y.append("Hyperelastic-Gent-CR-Budday2017")
    data_C.append(c)
    data_A.append(Common.WM)
    data_M.append(Common.HYP)

    # [67]
    model = Nonlinear.Budday2017d("Gent BG [Budday2017]", 1.01 * kPa, 0.08, 0.52 * kPa, 0.18)
    model.Compute()
    l, W, dW, H, sigma, P, offset = model.Evaluate(MechanicalTests.Uniaxial, [0.9, 1.1])
    data_l = np.concatenate((data_l, l.reshape(1, s)), axis=0)
    data_W = np.concatenate((data_W, W.reshape(1, s)), axis=0)
    data_dW = np.concatenate((data_dW, dW[0].reshape(1, s)), axis=0)
    data_H = np.concatenate((data_H, H[0, 0].reshape(1, s)), axis=0)
    data_sigma = np.concatenate((data_sigma, sigma[0].reshape(1, s)), axis=0)
    data_P = np.concatenate((data_P, P.reshape(1, p)), axis=0)
    data_offset.append(offset)
    data_Y.append("Hyperelastic-Gent-BG-Budday2017")
    data_C.append(c)
    data_A.append(Common.GM)
    data_M.append(Common.HYP)

    model = Nonlinear.Budday2017d("Gent C [Budday2017]", 2.01 * kPa, 0.08, 0.92 * kPa, 0.08)
    model.Compute()
    l, W, dW, H, sigma, P, offset = model.Evaluate(MechanicalTests.Uniaxial, [0.9, 1.1])
    data_l = np.concatenate((data_l, l.reshape(1, s)), axis=0)
    data_W = np.concatenate((data_W, W.reshape(1, s)), axis=0)
    data_dW = np.concatenate((data_dW, dW[0].reshape(1, s)), axis=0)
    data_H = np.concatenate((data_H, H[0, 0].reshape(1, s)), axis=0)
    data_sigma = np.concatenate((data_sigma, sigma[0].reshape(1, s)), axis=0)
    data_P = np.concatenate((data_P, P.reshape(1, p)), axis=0)
    data_offset.append(offset)
    data_Y.append("Hyperelastic-Gent-C-Budday2017")
    data_C.append(c)
    data_A.append(Common.GM)
    data_M.append(Common.HYP)

    ''' Budday Ogden [69] '''
    # model = Nonlinear.Budday2017e("Mod 1-term Ogden CC [Budday2017]", 0.47 * kPa, -11.4, 0.33*kPa, -25.6)
    model = Nonlinear.Budday2017e("Mod 1-term Ogden CC [Budday2017]", 0.43 * kPa, -22.8, 0.35 * kPa, -26.6)
    model.Compute()
    l, W, dW, H, sigma, P, offset = model.Evaluate(MechanicalTests.Uniaxial, [0.9, 1.1])
    data_l = np.concatenate((data_l, l.reshape(1, s)), axis=0)
    data_W = np.concatenate((data_W, W.reshape(1, s)), axis=0)
    data_dW = np.concatenate((data_dW, dW[0].reshape(1, s)), axis=0)
    data_H = np.concatenate((data_H, H[0, 0].reshape(1, s)), axis=0)
    data_sigma = np.concatenate((data_sigma, sigma[0].reshape(1, s)), axis=0)
    data_P = np.concatenate((data_P, P.reshape(1, p)), axis=0)
    data_offset.append(offset)
    data_Y.append("Hyperelastic-Ogden1-CC-Budday2017")
    data_C.append(c)
    data_A.append(Common.WM)
    data_M.append(Common.HYP)

    # model = Nonlinear.Budday2017e("Mod 1-term Ogden CR [Budday2017]", 0.86 * kPa, -19.9, 0.58 * kPa, -29.2)
    model = Nonlinear.Budday2017e("Mod 1-term Ogden CR [Budday2017]", 0.85 * kPa, -20.5, 0.61 * kPa, -30.5)
    model.Compute()
    l, W, dW, H, sigma, P, offset = model.Evaluate(MechanicalTests.Uniaxial, [0.9, 1.1])
    data_l = np.concatenate((data_l, l.reshape(1, s)), axis=0)
    data_W = np.concatenate((data_W, W.reshape(1, s)), axis=0)
    data_dW = np.concatenate((data_dW, dW[0].reshape(1, s)), axis=0)
    data_H = np.concatenate((data_H, H[0, 0].reshape(1, s)), axis=0)
    data_sigma = np.concatenate((data_sigma, sigma[0].reshape(1, s)), axis=0)
    data_P = np.concatenate((data_P, P.reshape(1, p)), axis=0)
    data_offset.append(offset)
    data_Y.append("Hyperelastic-Ogden1-CR-Budday2017")
    data_C.append(c)
    data_A.append(Common.WM)
    data_M.append(Common.HYP)

    # [71]
    # model = Nonlinear.Budday2017e("Mod 1-term Ogden BG [Budday2017]", 0.84 * kPa, -12.5, 0.64 * kPa, -30.0)
    model = Nonlinear.Budday2017e("Mod 1-term Ogden BG [Budday2017]", 0.83 * kPa, -15.5, 0.65 * kPa, -32.5)
    model.Compute()
    l, W, dW, H, sigma, P, offset = model.Evaluate(MechanicalTests.Uniaxial, [0.9, 1.1])
    data_l = np.concatenate((data_l, l.reshape(1, s)), axis=0)
    data_W = np.concatenate((data_W, W.reshape(1, s)), axis=0)
    data_dW = np.concatenate((data_dW, dW[0].reshape(1, s)), axis=0)
    data_H = np.concatenate((data_H, H[0, 0].reshape(1, s)), axis=0)
    data_sigma = np.concatenate((data_sigma, sigma[0].reshape(1, s)), axis=0)
    data_P = np.concatenate((data_P, P.reshape(1, p)), axis=0)
    data_offset.append(offset)
    data_Y.append("Hyperelastic-Ogden1-BG-Budday2017")
    data_C.append(c)
    data_A.append(Common.GM)
    data_M.append(Common.HYP)

    # model = Nonlinear.Budday2017e("Mod 1-term Ogden C [Budday2017]", 1.63 * kPa, -16.5, 1.16 * kPa, -38.9)
    model = Nonlinear.Budday2017e("Mod 1-term Ogden C [Budday2017]", 1.61 * kPa, -16.6, 1.20 * kPa, -43.6)
    model.Compute()
    l, W, dW, H, sigma, P, offset = model.Evaluate(MechanicalTests.Uniaxial, [0.9, 1.1])
    data_l = np.concatenate((data_l, l.reshape(1, s)), axis=0)
    data_W = np.concatenate((data_W, W.reshape(1, s)), axis=0)
    data_dW = np.concatenate((data_dW, dW[0].reshape(1, s)), axis=0)
    data_H = np.concatenate((data_H, H[0, 0].reshape(1, s)), axis=0)
    data_sigma = np.concatenate((data_sigma, sigma[0].reshape(1, s)), axis=0)
    data_P = np.concatenate((data_P, P.reshape(1, p)), axis=0)
    data_offset.append(offset)
    data_Y.append("Hyperelastic-Ogden1-C-Budday2017")
    data_C.append(c)
    data_A.append(Common.GM)
    data_M.append(Common.HYP)

    PlotMaterial.Groups("Hyperelastic Nonlinear", stretches, data_l, data_W, data_dW, data_H, data_sigma, data_Y)

    return data_l, data_W, data_dW, data_H, data_sigma, data_P, data_offset, data_Y, data_C, data_A, data_M