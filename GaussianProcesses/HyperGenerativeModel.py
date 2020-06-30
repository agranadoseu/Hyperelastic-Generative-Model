"""
Based on this example in GPy
https://nbviewer.jupyter.org/github/SheffieldML/notebook/blob/master/GPy/multiple%20outputs.ipynb
"""


import numpy as np
import GPy
import climin
from sklearn.preprocessing import MinMaxScaler
import matplotlib
import matplotlib.pyplot as plt
from matplotlib.ticker import StrMethodFormatter

from Hyperelasticity import LiteratureData
from Optimisation.HyperMetaModel import HyperMetaModel

# parameters
N = 100             # discretisation of W
offset = 0.0001     # log function to avoid zero


class HyperGenerativeModel:
    def __init__(self, process_data=False):

        # sample size
        self.S = 501

        # colours used for plots
        self.colourcat = ['#28e0f2', '#34bac5', '#5e80bc', '#7061db', '#bfed50', '#b5d568', '#6caf7f', '#008486', '#d4d6bd', '#fcaf29', '#e8b3ab', '#e16a69']

        # load data from papers
        if process_data is False:
            self.lambdas, self.L, self.W, self.dW, self.H, self.sigma, self.Y, self.C, self.A, self.M = LiteratureData.Load()
        else:
            self.lambdas, self.L, self.W, self.dW, self.H, self.sigma, self.Y, self.C, self.A, self.M = LiteratureData.Process()

        # convert W to log space (avoid zero)
        self.logW = np.log(self.W + offset)

        # select cases per task
        self.select_cases_per_task()
        # self.plot_data()

        # format data
        self.GX, self.Gy, self.Nx = self.format_data()

    def select_cases_per_task(self):
        self.gm_e = self.select_cases(self.A, self.M, [0], [0])  # 0
        self.wm_e = self.select_cases(self.A, self.M, [1], [0])  # 1
        self.nh_e = self.select_cases(self.A, self.M, [2], [0])  # 2
        self.ab_e = self.select_cases(self.A, self.M, [3], [0])  # 3

        self.gm_l = self.select_cases(self.A, self.M, [0], [1])  # 4
        self.wm_l = self.select_cases(self.A, self.M, [1], [1])  # 5
        self.nh_l = self.select_cases(self.A, self.M, [2], [1])  # 6
        self.ab_l = self.select_cases(self.A, self.M, [3], [1])  # 7

        self.gm_h = self.select_cases(self.A, self.M, [0], [2])  # 8
        self.wm_h = self.select_cases(self.A, self.M, [1], [2])  # 9
        self.nh_h = self.select_cases(self.A, self.M, [2], [2])  # 10
        self.ab_h = self.select_cases(self.A, self.M, [3], [2])  # 11

        print("Cases:")
        print("GM/E", self.gm_e, "WM/E", self.wm_e, "NH/E", self.nh_e, "AB/E", self.ab_e)
        print("GM/L", self.gm_l, "WM/L", self.wm_l, "NH/L", self.nh_l, "AB/L", self.ab_l)
        print("GM/H", self.gm_h, "WM/H", self.wm_h, "NH/H", self.nh_h, "AB/H", self.ab_h)

    def select_cases(self, _A, _M, _anatomy=[], _models=[]):
        _N = len(_A)
        _cases = []

        for i in range(_N):
            a = _A[i]
            m = _M[i]
            if a in _anatomy and m in _models:
                _cases.append(i)

        return np.asarray(_cases)

    def plot_cases(self, _cases):
        fig = plt.figure(1)
        fig.suptitle("Data input", fontsize=14)
        for i in range(_cases.shape[0]):
            idx = _cases[i]

            axis = plt.subplot(121)
            plt.plot(self.L[idx, :], self.W[idx, :], color=self.colourcat[self.M[idx] * 4 + self.A[idx]])
            # plt.text(L[idx,0], W[idx,0], Y[idx])

            axis = plt.subplot(122)
            plt.plot(self.L[idx, :], self.logW[idx, :], color=self.colourcat[self.M[idx] * 4 + self.A[idx]])
            # plt.text(L[idx, 0], logW[idx, 0], Y[idx])
        plt.show()

    def plot_data(self):
        # elastography
        mre_cases = np.asarray([], dtype=np.int32)
        mre_cases = np.append(mre_cases, self.gm_e)
        mre_cases = np.append(mre_cases, self.wm_e)
        mre_cases = np.append(mre_cases, self.nh_e)
        mre_cases = np.append(mre_cases, self.ab_e)
        self.plot_cases(mre_cases)

        # linear
        lin_cases = np.asarray([], dtype=np.int32)
        lin_cases = np.append(lin_cases, self.gm_l)
        lin_cases = np.append(lin_cases, self.wm_l)
        lin_cases = np.append(lin_cases, self.nh_l)
        lin_cases = np.append(lin_cases, self.ab_l)
        self.plot_cases(lin_cases)

        # hyperelastic
        hyp_cases = np.asarray([], dtype=np.int32)
        hyp_cases = np.append(hyp_cases, self.gm_h)
        hyp_cases = np.append(hyp_cases, self.wm_h)
        hyp_cases = np.append(hyp_cases, self.nh_h)
        hyp_cases = np.append(hyp_cases, self.ab_h)
        self.plot_cases(hyp_cases)

    def plot_data_literature(self):
        # from literature
        studies = ["Brain Surgery Simulation", "Simulation of Abnormal Developments", "Pathological Region Growing",
                   "MRI Registration", "Brain-shift Compensation", "MR Elastography", "Hyperelastic Nonlinear"]
        cases = [2, 2, 8, 2, 12, 12, 35]

        # (73, 100)
        print('Literature [all]')
        print('shape L = ', self.L.shape)
        print('shape W = ', self.W.shape)
        matplotlib.rcParams.update({'font.size': 12})

        index = 0
        for i in range(len(cases)):
            stretches = self.L[index:index+cases[i],:]
            potential = self.W[index:index+cases[i],:]
            labels = self.Y[index:index + cases[i]]

            print('     Literature [' + studies[i] + '] with ' + str(cases[i]) + ' cases')
            print('     shape L = ', stretches.shape)
            print('     shape W = ', potential.shape)

            fig = plt.figure(1)
            plt.plot(np.transpose(stretches), np.transpose(potential))
            for j in range(potential.shape[0]):
                # plt.text(stretches[j, 0], potential[j, 0], labels[j]+'['+str(index+j)+']')
                plt.text(stretches[j, 0], potential[j, 0], str(index + j), fontsize=12)
            plt.title(studies[i], fontsize=12)
            plt.xlabel("Stretches (lambda)")
            plt.ylabel("Strain energy (W)")
            # axis.set_xlim([0.6, 1.4])
            plt.xlim([0.99*min(map(min, stretches)), 1.01*max(map(max, stretches))])
            plt.ylim([0, 1.1*max(map(max, potential))])
            plt.gca().xaxis.set_major_formatter(StrMethodFormatter('{x:,.2f}'))
            plt.show()

            index += cases[i]

    def create_data_group(self, _gc, _X, _y):
        size = len(_gc)
        print("CreateGroup: ", _gc, size)
        GroupX = np.zeros((100 * size, 1))
        GroupY = np.zeros((100 * size, 1))

        for i in range(size):
            _gx = _X[_gc[i] * 100: _gc[i] * 100 + 100, 0]
            _gy = _y[_gc[i] * 100: _gc[i] * 100 + 100, 0]
            GroupX[i * 100:i * 100 + 100, 0] = _gx
            GroupY[i * 100:i * 100 + 100, 0] = _gy

        return GroupX, GroupY, size

    def format_data(self):
        # X has 4 columns: lambda (0), task (1), anatomy (2), model (3)
        # y has one column: logW (0)
        G = np.ones((self.L.shape[0], self.L.shape[1]))
        GA = np.ones((self.L.shape[0], self.L.shape[1]))
        GM = np.ones((self.L.shape[0], self.L.shape[1]))
        for i in range(self.L.shape[0]):
            G[i, :] = self.M[i] * 4 + self.A[i]
            GA[i, :] = self.A[i]
            GM[i, :] = self.M[i]
        X = np.zeros((self.L.shape[0] * self.L.shape[1], 4))
        y = np.zeros((self.W.shape[0] * self.W.shape[1], 1))
        X[:, 0] = np.transpose(self.L.reshape(-1, self.L.shape[0] * self.L.shape[1]))[:, 0]
        X[:, 1] = np.transpose(G.reshape(-1, G.shape[0] * G.shape[1]))[:, 0]
        X[:, 2] = np.transpose(GA.reshape(-1, GA.shape[0] * GA.shape[1]))[:, 0]
        X[:, 3] = np.transpose(GM.reshape(-1, GM.shape[0] * GM.shape[1]))[:, 0]
        y[:, 0] = np.transpose(self.logW.reshape(-1, self.logW.shape[0] * self.logW.shape[1]))[:, 0]

        # group
        GX0, GY0, N0 = self.create_data_group(self.gm_e, X, y)    # np.asarray([27 29 31 33 36])
        GX1, GY1, N1 = self.create_data_group(self.wm_e, X, y)    # np.asarray([26 28 30 32 34])
        GX2, GY2, N2 = self.create_data_group(self.nh_e, X, y)    # np.asarray([37])
        GX3, GY3, N3 = self.create_data_group(self.ab_e, X, y)    # np.asarray([35])
        GX4, GY4, N4 = self.create_data_group(self.gm_l, X, y)    # np.asarray([8])
        GX5, GY5, N5 = self.create_data_group(self.wm_l, X, y)    # np.asarray([7])
        GX6, GY6, N6 = self.create_data_group(self.nh_l, X, y)    # np.asarray([ 0  1  4  5 12 13 14 15 16 17 18 19 21 22 24 25])
        GX7, GY7, N7 = self.create_data_group(self.ab_l, X, y)    # np.asarray([ 2  3  9 10 11 20 23])
        GX8, GY8, N8 = self.create_data_group(self.gm_h, X, y)    # np.asarray([55 56 59 60 63 64 67 68 71 72])
        GX9, GY9, N9 = self.create_data_group(self.wm_h, X, y)    # np.asarray([53 54 57 58 61 62 65 66 69 70])
        GX10, GY10, N10 = self.create_data_group(self.nh_h, X, y) # np.asarray([38 39 40 41 42 43 44 45 46 47 48 49 50 51 52])
        GX11, GY11, N11 = self.create_data_group(self.ab_h, X, y) # np.asarray([6])

        GX = [GX0, GX1, GX2, GX3, GX4, GX5, GX6, GX7, GX8, GX9, GX10, GX11]
        Gy = [GY0, GY1, GY2, GY3, GY4, GY5, GY6, GY7, GY8, GY9, GY10, GY11]
        Nx = [N0, N1, N2, N3, N4, N5, N6, N7, N8, N9, N10, N11]
        # print("GX: ", GX)
        # print("Gy: ", Gy)
        print("Nx: ", Nx)

        return GX, Gy, Nx

    def create_model_gplog12(self):
        print("GPlog12 :: X1={} y1={}".format(len(self.GX), len(self.Gy)))

        # create kernel
        self.K = GPy.kern.Matern32(1)
        self.icm = GPy.util.multioutput.ICM(input_dim=1, num_outputs=12, kernel=self.K)

        # create model
        self.m = GPy.models.GPCoregionalizedRegression(self.GX, self.Gy, kernel=self.icm)
        self.m['.*Mat32.var'].constrain_fixed(1.)  # For this kernel, B.kappa encodes the variance now.
        print('kernel = {}\nicm = {}\nmodel = {}'.format(self.K, self.icm, self.m))

        # regress
        self.m.optimize(messages=1, max_iters=5e5)

        # save data
        np.save('./data/GP_icm12_params.npy', self.m.param_array)
        np.save('./data/GP_icm12_X.npy', self.GX)
        np.save('./data/GP_icm12_y.npy', self.Gy)
        np.save('./data/GP_icm12_n.npy', self.Nx)

        # print results
        print("12 tasks")
        print("Optimised model", self.m)
        print("\n\nB={} \n\nW={} \n\nK={}".format(self.m.ICM.B, self.m.ICM.B.W, self.m.ICM.B.kappa))

        # compute covariance matrix
        W = self.m.ICM.B.W.values
        k = self.m.ICM.B.kappa.values
        B = W*W.transpose() + k*np.eye(12)
        np.save('./data/GP_icm12_B.npy', self.m.ICM.B)
        np.save('./data/GP_icm12_W.npy', self.m.ICM.B.W)
        np.save('./data/GP_icm12_k.npy', self.m.ICM.B.kappa)

    def load_model(self):
        # params = np.load('./data/great/GP_icm12_params.npy')
        # X = np.load('./data/great/GP_icm12_X.npy')
        # y = np.load('./data/great/GP_icm12_y.npy')
        # n = np.load('./data/great/GP_icm12_n.npy')

        params = np.load('C:/UCL/PhysicsSimulation/Python/Biomechanics/GaussianProcesses/data/new1/GP_icm12_params.npy', allow_pickle=True)
        X = np.load('C:/UCL/PhysicsSimulation/Python/Biomechanics/GaussianProcesses/data/new1/GP_icm12_X.npy', allow_pickle=True)
        y = np.load('C:/UCL/PhysicsSimulation/Python/Biomechanics/GaussianProcesses/data/new1/GP_icm12_y.npy', allow_pickle=True)
        n = np.load('C:/UCL/PhysicsSimulation/Python/Biomechanics/GaussianProcesses/data/new1/GP_icm12_n.npy', allow_pickle=True)

        _r = 1
        K = GPy.kern.Matern32(1)
        icm = GPy.util.multioutput.ICM(input_dim=1, num_outputs=12, kernel=K)

        m = GPy.models.GPCoregionalizedRegression(X, y, kernel=icm)

        m.update_model(False)
        m.initialize_parameter()
        m[:] = params
        m.update_model(True)
        # print("m=", m)

        return m, X, y, n

    def plot_tasks_each(self, _model, _X, _y, _Nx):
        print("_X", len(_X))
        print("_y", len(_y))
        fig = plt.figure(1)
        ss = 0
        for i in range(12):
            doLegend = False
            if i == 0:
                doLegend = True

            c = '#34bac5'  # blue
            cd = '#7061db'
            xm = np.linspace(0.95, 1.05, 101)[:, None]
            if int(i / 4) == 1:
                c = '#b5d568'  # green
                cd = '#008486'
                xm = np.linspace(0.9, 1.1, 101)[:, None]
            if int(i / 4) == 2:
                c = '#e8b3ab'  # red
                cd = '#e16a69'
                if int(i % 4) == 0:
                    xm = np.linspace(0.9, 1.1, 101)[:, None]
                if int(i % 4) == 1:
                    xm = np.linspace(0.9, 1.1, 101)[:, None]
                if int(i % 4) == 2:
                    xm = np.linspace(0.6, 1.4, 101)[:, None]
                if int(i % 4) == 3:
                    xm = np.linspace(0.8, 1.2, 101)[:, None]

            axis = plt.subplot(3, 4, i + 1)
            _model.plot(plot_limits=(0.6, 1.4), fixed_inputs=[(1, i)], which_data_rows=slice(ss, ss + 100 * _Nx[i]), ax=axis, legend=doLegend)
            ss += 100 * _Nx[i]

            # _model.plot(plot_limits=(0.6, 1.4), fixed_inputs=[(1, i)], plot_data=False, ax=axis, legend=doLegend)
            # plt.plot(_X[i], _y[i], 'o', markersize=4, alpha=0.3, color='green')

            # xm = np.linspace(0.6, 1.4, 101)[:, None]
            xm = np.c_[xm, np.ones(xm.shape[0]) * i]
            noise_dict = {'output_index': xm[:, 1:].astype(int)}
            # print("xm for this region", xm.shape, xm)
            # mu, var = m._raw_predict(xm)
            mu, var = _model.predict(xm, Y_metadata=noise_dict)
            # print("mu=", mu)
            # to confirm
            # plt.plot(xm, mu, '-', markersize=4, alpha=1.0, color='black')
            # plt.plot(xm, mu+2.*np.sqrt(var), '-.', markersize=4, alpha=1, color='black')
            # plt.plot(xm, mu-2.*np.sqrt(var), '-.', markersize=4, alpha=1, color='black')

            axis.set_xlim([min(xm[:, 0]), max(xm[:, 0])])

        plt.xlabel('stretches')
        plt.ylabel('log W')
        plt.show()

    def plot_tasks_each_W(self, _model, _X, _y, _Nx, _metamodel):
        fig = plt.figure(1)
        ss = 0
        for i in range(12):
            doLegend = False
            if i == 0:
                doLegend = True

            axis = plt.subplot(3, 4, i + 1)
            # _model.plot(plot_limits=(0.6, 1.4), fixed_inputs=[(1, i)], which_data_rows=slice(ss, ss + 100 * _Nx[i]), ax=axis, legend=doLegend)
            ss += 100 * _Nx[i]

            c = '#34bac5'  # blue
            cd = '#7061db'
            xm = np.linspace(0.95, 1.05, 101)[:, None]
            if int(i / 4) == 1:
                c = '#b5d568'  # green
                cd = '#008486'
                xm = np.linspace(0.9, 1.1, 101)[:, None]
            if int(i / 4) == 2:
                c = '#e8b3ab'  # red
                cd = '#e16a69'
                if int(i % 4) == 0:
                    xm = np.linspace(0.9, 1.1, 101)[:, None]
                if int(i % 4) == 1:
                    xm = np.linspace(0.9, 1.1, 101)[:, None]
                if int(i % 4) == 2:
                    xm = np.linspace(0.6, 1.4, 101)[:, None]
                if int(i % 4) == 3:
                    xm = np.linspace(0.8, 1.2, 101)[:, None]

            #plt.plot(_X[i], data_inverseTransform(_y[i]), 'o', markersize=4, alpha=0.5, color=c)
            plt.plot(_X[i], np.exp(_y[i])-0.001, 'o', markersize=4, alpha=0.5, color=c)

            xm = np.c_[xm, np.ones(xm.shape[0]) * i]
            noise_dict = {'output_index': xm[:, 1:].astype(int)}
            # print("xm for this region", xm.shape, xm)
            # mu, var = m._raw_predict(xm)
            mu, var = _model.predict(xm, Y_metadata=noise_dict)
            # print("mu=", mu)

            print('\nTask={}'.format(i+1))
            # w_mm, cost, xt, yt_log, yt = fit_meta_model(xm[:, 0], mu[:, 0])
            w_mm, cost, xt, yt_log, yt = _metamodel.optimise(xm[:, 0], mu[:, 0])
            plt.plot(xt, yt, '-', markersize=4, alpha=1.0, color='black')

            # sweep
            variances = np.linspace(-2.0, 2.0, int(10) + 1)
            for v in variances:
                mu[:, 0] + v * np.sqrt(var[:, 0])
                w_mm, cost, xt, yt_log, yt = _metamodel.optimise(xm[:, 0], mu[:, 0] + v * np.sqrt(var[:, 0]))
                plt.plot(xt, yt, '-', markersize=4, alpha=0.5, color='black')

            # CI
            plt.plot(xm, np.exp(mu)-0.001, '-', markersize=6, alpha=1.0, color=cd)
            plt.plot(xm, np.exp(mu + 2. * np.sqrt(var))-0.001, '-.', markersize=6, alpha=1, color=cd)
            plt.plot(xm, np.exp(mu - 2. * np.sqrt(var))-0.001, '-.', markersize=6, alpha=1, color=cd)

            # axis.set_xlim([0.6, 1.4])
            axis.set_xlim([min(xm[:, 0]), max(xm[:, 0])])
            axis.set_ylim([0, max(map(max, np.exp(_y[i])-0.001))])

        plt.xlabel('stretches')
        plt.ylabel('log W')
        plt.show()


    def plot_cases_with_metamodels(self, _cases, _wider_L, _wider_W, _metamodels):
        matplotlib.rcParams.update({'font.size': 16})
        # xm = np.linspace(0.00001, 5.0, 100)
        xm = np.linspace(0.9, 1.1, 100)
        fig = plt.figure(1)

        # self.colourcat[self.M[i]*4 + self.A[i]]
        for i in _cases:
            markers = ['*',':','--','-']
            for m in range(len(_metamodels)):
                w_mm, cost, xt, yt_log, yt = _metamodels[m].optimise(xm, self.logW[i])
                plt.plot(xt, yt, markers[m], label=_metamodels[m].name+' c={}'.format(np.around(cost,decimals=2)), linewidth=3, markersize=4, alpha=1.0, color='black')

            plt.plot(_wider_L, _wider_W, '-', label='Extrapolated model', linewidth=3, markersize=4, alpha=0.5, color='darkorchid')
            plt.plot(self.L[i], np.exp(self.logW[i]) - 0.001, 'o', label=self.Y[i], markersize=8, alpha=0.3, color='deepskyblue')

        plt.xlabel('stretches')
        plt.ylabel('W')
        handles, labels = plt.gca().get_legend_handles_labels()
        order = [5, 4, 0, 1, 2, 3]
        plt.legend([handles[idx] for idx in order], [labels[idx] for idx in order], loc='upper right')
        # plt.legend(loc='upper right')  # 'upper center'
        # plt.ylim(-.2, 5.2)
        plt.ylim(0, 10000)  # 10000 1000000
        plt.show()


# hyper_gen_model = HyperGenerativeModel(process_data=True)
# hyper_gen_model.create_model_gplog12()