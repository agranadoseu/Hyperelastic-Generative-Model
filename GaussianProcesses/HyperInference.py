"""
This class uses a metamodel to fit samples from GP distributions
It also assist to create the simulation environment for sampling
"""

import os
import sys
import math
import numpy as np
import GPy
from scipy.optimize import least_squares
import matplotlib.pyplot as plt

from Hyperelasticity.utils import Common
from Hyperelasticity.utils import MechanicalTests
from Hyperelasticity import Linear, Nonlinear
from GaussianProcesses.HyperGenerativeModel import HyperGenerativeModel
from Optimisation.NeoHookeanMetaModel import NeoHookeanMetaModel, NeoHookeanPenaltyMetaModel
from Optimisation.MooneyRivlinMetaModel import MooneyRivlinMetaModel, MooneyRivlinPenaltyMetaModel
from Optimisation.OgdenMetaModel import OgdenMetaModel


class HyperInference:

    metamodel = 0       # HyperMetaModel
    hyper_gen_model = 0 # HyperGenerativeModel
    mGP = 0 # GP model
    X = 0   # data
    y = 0   # data
    Nx = 0  # data

    metamodel_nh = 0
    metamodel_mr = 0
    metamodel_o1_l = 0
    metamodel_o1_h = 0
    metamodel_o2 = 0
    metamodel_o3 = 0

    def __init__(self, metamodel):
        self.load_GP()
        self.create_metamodel(metamodel)

    def load_GP(self):
        self.hyper_gen_model = HyperGenerativeModel()
        self.mGP, self.X, self.y, self.Nx = self.hyper_gen_model.load_model()

    def plot_GP(self):
        # plot data
        self.hyper_gen_model.plot_data()
        self.hyper_gen_model.plot_data_literature()

        # plot tasks
        # plot_tasks(mGP, Nx)
        self.hyper_gen_model.plot_tasks_each(self.mGP, self.X, self.y, self.Nx)
        self.hyper_gen_model.plot_tasks_each_W(self.mGP, self.X, self.y, self.Nx, self.metamodel)

    def plot_metamodel_all(self):
        # self.hyper_gen_model.select_cases_per_task()
        # GM/E [27 29 31 33 36] WM/E [26 28 30 32 34] NH/E [37] AB/E [35]
        # GM/L [8] WM/L [7] NH/L [ 0  1  4  5 12 13 14 15 16 17 18 19 21 22 24 25] AB/L [ 2  3  9 10 11 20 23]
        # GM/H [55 56 59 60 63 64 67 68 71 72] WM/H [53 54 57 58 61 62 65 66 69 70] NH/H [38 39 40 41 42 43 44 45 46 47 48 49 50 51 52] AB/H [6]
        c_h_gm, c_h_wm, c_h_nh, c_h_ab = np.asarray([71]), np.asarray([58]), np.asarray([45]), np.asarray([6])
        c_all = np.concatenate((c_h_gm, c_h_wm, c_h_nh, c_h_ab), axis=0)
        # self.hyper_gen_model.plot_cases(c_h_gm)
        # self.hyper_gen_model.plot_cases(c_h_wm)
        # self.hyper_gen_model.plot_cases(c_h_nh)
        # self.hyper_gen_model.plot_cases(c_h_ab)
        # self.hyper_gen_model.plot_cases(c_all)

        kPa = Common.kPa
        # mm_l = [self.metamodel_nh, self.metamodel_mr, self.metamodel_o1_l]
        mm_l = [self.metamodel_nh, self.metamodel_mr, self.metamodel_o1_l, self.metamodel_o1_h]
        mm_h = [self.metamodel_nh, self.metamodel_mr, self.metamodel_o1_h]

        print('\n\n>>>>> Neo-Hookean')
        model = Nonlinear.Budday2017a("Neo-Hookean CC [Budday2017]", 0.99 * kPa, 0.29 * kPa)
        model.Compute()
        l, W, dW, H, sigma, P, offset = model.Evaluate(MechanicalTests.Uniaxial, [0.001, 5.0])
        self.hyper_gen_model.plot_cases_with_metamodels(np.asarray([53]), l, W, mm_l)

        print('\n\n>>>>> Mooney-Rivlin')
        model = Nonlinear.Budday2017b("Mooney-Rivlin BG [Budday2017]", 1.27 * kPa, 0.0 * kPa, 0.57 * kPa, 0.29 * kPa)
        model.Compute()
        l, W, dW, H, sigma, P, offset = model.Evaluate(MechanicalTests.Uniaxial, [0.001, 5.0])
        self.hyper_gen_model.plot_cases_with_metamodels(np.asarray([59]), l, W, mm_l)

        print('\n\n>>>>> Ogden 1-term')
        model = Nonlinear.Mihai2017a("Ogden 1-term [Mihai2017]", -0.0939 * kPa, -4.0250)
        model.Compute()
        l, W, dW, H, sigma, P, offset = model.Evaluate(MechanicalTests.Uniaxial, [0.001, 5.0])
        self.hyper_gen_model.plot_cases_with_metamodels(np.asarray([38]), l, W, mm_l)

        print('\n\n>>>>> Ogden 2-terms')
        model = Nonlinear.Budday2017e("Mod 1-term Ogden BG [Budday2017]", 0.83 * kPa, -15.5, 0.65 * kPa, -32.5)
        model.Compute()
        l, W, dW, H, sigma, P, offset = model.Evaluate(MechanicalTests.Uniaxial, [0.001, 5.0])
        self.hyper_gen_model.plot_cases_with_metamodels(np.asarray([71]), l, W, mm_l)

        print('\n\n>>>>> Ogden 3-terms')
        model = Nonlinear.Mihai2015e("Ogden 3-term [Mihai2015]", 3, [-3.543 * kPa, -2.723 * kPa, 0.654 * kPa], [1, -1, 2])
        model.Compute()
        l, W, dW, H, sigma, P, offset = model.Evaluate(MechanicalTests.Uniaxial, [0.001, 5.0])
        self.hyper_gen_model.plot_cases_with_metamodels(np.asarray([44]), l, W, mm_l)

        print('\n\n>>>>> Ogden 4-terms')
        model = Nonlinear.Mihai2015e("Ogden 4-term [Mihai2015]", 4, [-5.877 * kPa, -5.043 * kPa, 1.161 * kPa, 0.501 * kPa], [1, -1, 2, -2])
        model.Compute()
        l, W, dW, H, sigma, P, offset = model.Evaluate(MechanicalTests.Uniaxial, [0.001, 5.0])
        self.hyper_gen_model.plot_cases_with_metamodels(np.asarray([45]), l, W, mm_l)

        print('\n\n>>>>> Ogden 8-terms')
        model = Nonlinear.Mihai2015e("Ogden 8-term [Mihai2015]", 8, [-639.530 * kPa, -544.840 * kPa, 322.660 * kPa, 237.040 * kPa, -88.640 * kPa, -57.830 * kPa, 10.150 * kPa, 6.080 * kPa], [1, -1, 2, -2, 3, -3, 4, -4])
        model.Compute()
        l, W, dW, H, sigma, P, offset = model.Evaluate(MechanicalTests.Uniaxial, [0.001, 5.0])
        self.hyper_gen_model.plot_cases_with_metamodels(np.asarray([49]), l, W, mm_l)


    def plot_prediction(self, w, xm, mean, variance, ym_log, xt, yt_log, yt):
        fig = plt.figure(1)
        title_str = self.metamodel.name + ": w="
        for i in w:
            title_str += str(np.around(i,decimals=2)) + " "
        fig.suptitle(title_str, fontsize=14)
        axis = plt.subplot(121)
        plt.plot(xm, mean, '-', markersize=4, alpha=1.0, color='black')
        plt.vlines(xm, mean - 2. * np.sqrt(variance), mean + 2. * np.sqrt(variance), color='black', lw=2, alpha=0.05)
        plt.plot(xm, ym_log, 'o', markersize=6, label='predicted sample', alpha=0.3)
        plt.plot(xt, yt_log, label=self.metamodel.name)
        plt.xlabel("x")
        plt.ylabel("y")
        # plt.ylim(0, 4e7)
        plt.legend(loc='lower right')

        axis = plt.subplot(122)
        plt.plot(xm, np.exp(mean)-0.001, '-', markersize=4, alpha=1.0, color='black')
        plt.vlines(xm, np.exp(mean - 2. * np.sqrt(variance))-0.001,
                   np.exp(mean + 2. * np.sqrt(variance))-0.001, color='black', lw=2, alpha=0.05)
        plt.plot(xm, np.exp(ym_log)-0.001, 'o', markersize=6, label='predicted sample', alpha=0.3)
        plt.plot(xt, yt, label=self.metamodel.name)
        plt.xlabel("x")
        plt.ylabel("y")
        # axis.set_ylim([0, ym[0]])
        # axis.set_ylim([0, 70000])
        axis.set_xlim([min(xm), max(xm)])
        axis.set_ylim([0, max(np.exp(ym_log)-0.001)])
        plt.legend(loc='lower right')

        mng = plt.get_current_fig_manager()
        mng.window.state('zoomed')
        plt.show()

    def create_metamodel(self, metamodel):
        if metamodel == 'all':
            self.metamodel_nh = NeoHookeanMetaModel()
            self.metamodel_mr = MooneyRivlinMetaModel()
            self.metamodel_o1_l = OgdenMetaModel(N=1, aim='lower')
            self.metamodel_o1_h = OgdenMetaModel(N=1, aim='higher')
            # self.metamodel_o2 = OgdenMetaModel(N=2)
            # self.metamodel_o3 = OgdenMetaModel(N=3)
        elif metamodel == 'Neo-Hookean':
            self.metamodel = NeoHookeanMetaModel()
            # self.metamodel = NeoHookeanPenaltyMetaModel()
        elif metamodel == 'Mooney-Rivlin':
            self.metamodel = MooneyRivlinMetaModel()
            # self.metamodel = MooneyRivlinPenaltyMetaModel()
        elif metamodel == 'Ogden1':
            # self.metamodel = OgdenMetaModel(N=1)
            # self.metamodel = OgdenMetaModel(N=1, aim='lower')
            self.metamodel = OgdenMetaModel(N=1, aim='higher')
        elif metamodel == 'Ogden2':
            self.metamodel = OgdenMetaModel(N=2)
        elif metamodel == 'Ogden3':
            self.metamodel = OgdenMetaModel(N=3)

    def predict(self, model, l, a_idx, m_idx, f_std):
        tx = l[:, None]
        taskId = np.ones_like(tx) * float(m_idx * 4 + a_idx)
        # print("taskId = ", float(m_idx*4+a_idx), "m_idx=", m_idx, "a_idx=", a_idx)
        tx = np.hstack([tx, taskId])
        noise_dict = {'output_index': tx[:, 1:].astype(int)}
        # output_index = np.zeros((tx.shape[0],1)).astype(int)
        # output_index[:,0] = int(m_idx*4+a_idx)
        print("tx", tx.shape, tx)
        # print("output_index", output_index)
        # mu, var = model.predict(tx, Y_metadata={'output_index':output_index})  # fetch the posterior of f
        mu, var = model.predict(tx, Y_metadata=noise_dict)
        pred = mu[:, 0] + float(f_std) * np.sqrt(var[:, 0])

        return mu[:, 0], var[:, 0], pred

    def sample_one(self, anatomy, model, f_stddev):
        # mGP, X, y, Nx = load_GP()
        # plot_tasks(mGP, Nx)

        lambdas = np.linspace(0.95, 1.05, self.hyper_gen_model.S)
        if model == 1:
            lambdas = np.linspace(0.9, 1.1, self.hyper_gen_model.S)
        if model == 2:
            if anatomy == 0 or anatomy == 1:
                lambdas = np.linspace(0.9, 1.1, self.hyper_gen_model.S)
            if anatomy == 2:
                lambdas = np.linspace(0.6, 1.4, self.hyper_gen_model.S)
            if anatomy == 3:
                lambdas = np.linspace(0.8, 1.2, self.hyper_gen_model.S)

        mu, var, pred_log = self.predict(self.mGP, lambdas, anatomy, model, f_stddev)
        w_mm, cost, xt, yt_log, yt = self.metamodel.optimise(lambdas, pred_log)
        self.plot_prediction(w_mm, lambdas, mu, var, pred_log, xt, yt_log, yt)

        return w_mm, cost

    def sampleMany(self, anatomy, model, stddev):
        # mGP, X, Y, Nx = load_GP()
        # plot_tasks(mGP, Nx)

        lambdas = np.linspace(0.95, 1.05, self.hyper_gen_model.S)
        if model == 1:
            lambdas = np.linspace(0.9, 1.1, self.hyper_gen_model.S)
        if model == 2:
            if anatomy == 0 or anatomy == 1:
                lambdas = np.linspace(0.9, 1.1, self.hyper_gen_model.S)
            if anatomy == 2:
                lambdas = np.linspace(0.6, 1.4, self.hyper_gen_model.S)
            if anatomy == 3:
                lambdas = np.linspace(0.8, 1.2, self.hyper_gen_model.S)

        w = []
        c = []

        for sd in stddev:
            # mu, var, pred_log = predict(mGP, lambdas, anatomy, model, sd)
            mu, var, pred_log = self.predict(self.mGP, lambdas, anatomy, model, sd)
            w_mm, cost, xt, yt_log, yt = self.metamodel.optimise(lambdas, pred_log)

            # TODO important!!!
            # self.plot_prediction(w_mm, lambdas, mu, var, pred_log, xt, yt_log, yt)

            w.append(np.around(w_mm,decimals=8))
            c.append(np.around(cost,decimals=8))

        return w, c


def parse_state(filename, scale=1.0):
    '''
    Reads a list of node positions from either a reference or a deformed state
    :param file:
    :return: an array of 3D positions (nx4: id, x, y, z)
    '''
    nodes = []
    file = open("C:/UCL/PhysicsSimulation/Unity/EpiNav/Assets/Data/Validation/" + filename, "r")

    lines = file.readlines()
    for l in lines:
        values_str = l.rstrip().split(' ')
        values = [float(i)*scale for i in values_str]
        nodes.append(values)
    file.close()

    nodes = np.asarray(nodes, dtype=np.float)
    print('ParseState() ::. file={} has {} nodes'.format(filename, len(nodes)))
    return nodes


def compute_similarity_metric(reference, deformation):
    '''
    Computes similarity measure between a set of volumetric nodes
    :param reference:
    :param deformation:
    :return: RMSE
    '''
    ref_nodes = reference[:,1:4]
    def_nodes = deformation[:,1:4]

    # size
    N = 0
    if len(ref_nodes) != len(def_nodes):
        print('ERROR compute_similarity_metric() ::. state vectors have different sizes')
        return [-1,-1,-1]
    N = len(ref_nodes)

    # compute displacements
    du = ref_nodes - def_nodes
    u = np.linalg.norm(du, axis=1)
    max_u = np.max(u)
    sum_u = np.sum(u)

    rmse = math.sqrt(sum_u / N)

    return [rmse, max_u, sum_u]


if __name__ == '__main__':

    msg = "commnad line usage\n\n"
    msg += "python.exe GaussianProcesses\HyperInference.py [metamodel] [samplemode] [anatomy] [model] [stddev] [tissue] [case] [resolution] [time] [ventricles]\n"
    msg += "  - metamodel: all, Neo-Hookean, Mooney-Rivlin, Ogden1, Ogden2, Ogden3\n"
    msg += "  - samplemode: plot, one, many, simulation\n"
    msg += "  - anatomy: 0 (grey matter), 1 (white matter), 2 (healthy), 3 (abnormal)\n"
    msg += "  - model: 0 (MRE), 1 (Linear), 2 (Hyperelastic)\n"
    msg += "  - stddev: [-2.0, 2.0] or number of samples \n"
    msg += "  - tissue: st (single tissue), mt (multi-tissue) \n"
    msg += "  - case: name of case to process \n"
    msg += "  - volumetric resolution: coarse, half, fine \n"
    msg += "  - time: run simulation for each case in a number of seconds \n"
    msg += "  - ventricles: number of nodes \n"
    if len(sys.argv) != 11:
        print(msg)
        sys.exit(0)

    ''' Parse arguments '''
    # Metamodel to use
    meta_model = ''
    mm = ''
    if sys.argv[1] == 'all' or sys.argv[1] == 'Neo-Hookean' or sys.argv[1] == 'Mooney-Rivlin' or sys.argv[1] == 'Ogden1' or sys.argv[1] == 'Ogden2' or sys.argv[1] == 'Ogden3':
        meta_model = sys.argv[1]
        if meta_model == 'Neo-Hookean':
            mm = 'nh'
        elif meta_model == 'Mooney-Rivlin':
            mm = 'mr'
        elif meta_model == 'Ogden1':
            mm = 'o1'
    else:
        print('Metamodel <{}> is not supported'.format(sys.argv[1]))
        sys.exit(0)

    # Simulation to perform
    sample_mode = ''
    if sys.argv[2] == 'plot' or sys.argv[2] == 'one' or sys.argv[2] == 'many' or sys.argv[2] == 'simulation':
        sample_mode = sys.argv[2]
    else:
        print('Sample mode <{}> is not supported'.format(sys.argv[2]))
        sys.exit(0)

    anatomy = int(sys.argv[3])
    model = int(sys.argv[4])
    stddev = float(sys.argv[5])
    tissue = sys.argv[6]
    case = sys.argv[7]
    resolution = sys.argv[8]
    time = sys.argv[9]
    ventricles = int(sys.argv[10])

    # inference
    inference = HyperInference(meta_model)

    if meta_model == 'all':
        inference.plot_metamodel_all()
        sys.exit()

    if sample_mode == 'plot':
        inference.plot_GP()

    if sample_mode == 'one':
        # sample a specific one given one std dev
        value, cost = inference.sample_one(anatomy=anatomy, model=model, f_stddev=stddev)
        # res = np.asarray([stddev, value, cost])
        print(stddev, value, cost)

    if sample_mode == 'many':
        # sample many given a sweep of std deviations
        stddev = np.linspace(-2.0, 2.0, int(stddev)+1)
        value, cost = inference.sampleMany(anatomy=anatomy, model=model, stddev=stddev)
        stddev = np.asarray(stddev)[:,None]
        value = np.asarray(value)[:,None]
        cost = np.asarray(cost)[:,None]
        res = np.hstack([stddev, value, cost])
        print("\n".join(" ".join(map(str, line)) for line in res))

    if sample_mode=='simulation':

        # delete
        os.chdir("C:/UCL/PhysicsSimulation/Unity/EpiNav")
        print('os.getcwd() = ', os.getcwd())

        # sample a many given a sweep of std deviations
        stddev = np.linspace(-2.0, 2.0, int(stddev) + 1)

        # open reference state (mt is used for both mt and st experiments)
        rs_filename = '{}_{}_{}_{}_ref.txt'.format(case, resolution, tissue, mm)
        rs_nodes = parse_state(rs_filename, scale=100.0)

        # volume to use according to case and resolution
        if case == 'MNI':
            volume = 'Volume/GaussianProcess/{}/{}_simulation_{}_{}.1'.format(case, case, tissue, resolution)
        else:
            volume = 'Volume/GaussianProcess/{}/{}_simulation_{}_rest.1'.format(case, case, tissue)

        # save results
        results_filename = 'SimulationResults_{}_{}_{}_{}_{}_{}.txt'.format(case, resolution, tissue, mm, anatomy, model)
        res_file = open("C:/UCL/PhysicsSimulation/Unity/EpiNav/Assets/Data/Validation/" + results_filename, "w+")
        header = 'case,metamodel,method,anatomy_type_1,anatomy_type_2,stddev_1,stddev_2,num_params,a1,a2,b1,b2,cost_1,cost_2,rs,ds,rmse_u,max_u,sum_u,ven_rmse_u,ven_max_u,ven_sum_u,res_rmse_u,res_max_u,res_sum_u\n'
        res_file.write(header)

        experiment = 0

        if tissue == 'st':
            value, cost = inference.sampleMany(anatomy=anatomy, model=model, stddev=stddev)

            sample = 0
            for v in value:
                print("v={}".format(v))
                params_v = [-1, -1]

                # write Unity params file with value
                file = open("C:/UCL/PhysicsSimulation/Unity/EpiNav/Assets/Data/GP-" + meta_model.replace('-', '') + ".txt", "w+")
                num_params = '{}\n'.format(len(v))

                file.write(str(num_params))
                params = 'GM '
                for i in range(len(v)):
                    params_v[i] = v[i]
                    params += str(v[i]) + ' '
                params += '\n'
                file.write(params)
                params = 'WM 0 0\n'
                file.write(params)
                file.close()

                # run simulation
                ds_filename = '{}_{}_{}_{}_{}_{}_{}.txt'.format(case, resolution, tissue, mm, anatomy, model, experiment)

                if case == 'MNI':
                    # os.system('"C:/UCL/PhysicsSimulation/Unity/EpiNav/Validation/GPBrain{}.exe -t {} -vol {} -ds {}"'.format(mm.upper(), time, volume, ds_filename))

                    # open deformed state and simulation state, compute similarity metric
                    ds_nodes = parse_state(ds_filename, scale=100.0)
                    [rmse_u, max_u, sum_u] = compute_similarity_metric(rs_nodes, ds_nodes)

                    # save results in a different file (stddev, params, similarity)
                    res = '{},{},{},{},,{},,{},{},{},,,{},,{},{},{},{},{},0,0,0,0,0,0\n'.format(case, meta_model, model, anatomy, stddev[sample],
                                                                                                len(v), params_v[0], params_v[1], cost[sample],
                                                                                                rs_filename, ds_filename, rmse_u, max_u, sum_u)
                else:
                    os.system('"C:/UCL/PhysicsSimulation/Unity/EpiNav/ValidationResection/BrainResection_{}_{}.exe -t {} -vol {} -ds {}"'.format(case, mm.upper(), time, volume, ds_filename))

                    # open deformed state and simulation state, compute similarity metric
                    ds_nodes = parse_state(ds_filename, scale=100.0)
                    [rmse_u, max_u, sum_u] = compute_similarity_metric(rs_nodes, ds_nodes)
                    [ven_rmse_u, ven_max_u, ven_sum_u] = compute_similarity_metric(rs_nodes[0:ventricles,:], ds_nodes[0:ventricles,:])
                    [res_rmse_u, res_max_u, res_sum_u] = compute_similarity_metric(rs_nodes[ventricles:,:], ds_nodes[ventricles:,:])

                    # save results in a different file (stddev, params, similarity)
                    res = '{},{},{},{},,{},,{},{},{},,,{},,{},{},{},{},{},{},{},{},{},{},{}\n'.format(case, meta_model, model, anatomy, stddev[sample],
                                                                                                      len(v), params_v[0], params_v[1], cost[sample],
                                                                                                      rs_filename, ds_filename, rmse_u, max_u, sum_u,
                                                                                                      ven_rmse_u, ven_max_u, ven_sum_u,
                                                                                                      res_rmse_u, res_max_u, res_sum_u)
                res_file.write(res)

                sample += 1
                experiment += 1

            res_file.close()

            print('stddev[{}]={}'.format(len(stddev), stddev))
            print('value[{}]={}'.format(len(value), value))
            print('cost[{}]={}'.format(len(cost), cost))
            stddev = np.asarray(stddev)[:, None]
            value = np.asarray(value)[:, None]
            cost = np.asarray(cost)[:, None]
            # res = np.hstack([stddev, value_gm, cost_gm, value_wm, cost_wm])
            # print("\n".join(" ".join(map(str, line)) for line in res))

        if tissue == 'mt':
            value_gm, cost_gm = inference.sampleMany(anatomy=0, model=model, stddev=stddev)
            value_wm, cost_wm = inference.sampleMany(anatomy=1, model=model, stddev=stddev)

            gm_i = 0
            for v_gm in value_gm:
                wm_i = 0
                for v_wm in value_wm:
                    print("v_gm={} v_wm={}".format(v_gm, v_wm))
                    params_gm = [-1, -1]
                    params_wm = [-1, -1]

                    # write Unity params file with value
                    file = open("C:/UCL/PhysicsSimulation/Unity/EpiNav/Assets/Data/GP-"+meta_model.replace('-','')+".txt", "w+")
                    num_params = '{}\n'.format(len(v_gm))

                    file.write(str(num_params))
                    params = 'GM '
                    for i in range(len(v_gm)):
                        params_gm[i] = v_gm[i]
                        params += str(v_gm[i]) + ' '
                    params += '\n'
                    file.write(params)
                    params = 'WM '
                    for i in range(len(v_wm)):
                        params_wm[i] = v_wm[i]
                        params += str(v_wm[i]) + ' '
                    params += '\n'
                    file.write(params)
                    file.close()

                    # run simulation
                    # ds_filename = '{}_{}_{}_{}_{}.txt'.format(case, resolution, tissue, mm, experiment)
                    # os.system('"C:/UCL/PhysicsSimulation/Unity/EpiNav/Validation/GPBrain{}.exe -t {} -vol {} -ds {}"'.format(mm.upper(), time, volume, ds_filename))

                    # run simulation
                    ds_filename = '{}_{}_{}_{}_{}_{}_{}.txt'.format(case, resolution, tissue, mm, anatomy, model, experiment)

                    if case == 'MNI':
                        # os.system('"C:/UCL/PhysicsSimulation/Unity/EpiNav/Validation/GPBrain{}.exe -t {} -vol {} -ds {}"'.format(mm.upper(), time, volume, ds_filename))

                        # open deformed state and simulation state, compute similarity metric
                        ds_nodes = parse_state(ds_filename, scale=100.0)
                        [rmse_u, max_u, sum_u] = compute_similarity_metric(rs_nodes, ds_nodes)

                        # save results in a different file (stddev, params, similarity)
                        res = '{},{},{},0,1,{},{},{},{},{},{},{},{},{},{},{},{},{},{},0,0,0,0,0,0\n'.format(case, meta_model, model, stddev[gm_i], stddev[wm_i],
                                                                                                            len(v_gm), params_gm[0], params_gm[1], params_wm[0], params_wm[1],
                                                                                                            cost_gm[gm_i], cost_wm[wm_i],
                                                                                                            rs_filename, ds_filename,
                                                                                                            rmse_u, max_u, sum_u)
                    else:
                        os.system('"C:/UCL/PhysicsSimulation/Unity/EpiNav/ValidationResection/BrainResection_{}_{}.exe -t {} -vol {} -ds {}"'.format(case, mm.upper(), time, volume, ds_filename))

                        # open deformed state and simulation state, compute similarity metric
                        ds_nodes = parse_state(ds_filename, scale=100.0)
                        [rmse_u, max_u, sum_u] = compute_similarity_metric(rs_nodes, ds_nodes)
                        [ven_rmse_u, ven_max_u, ven_sum_u] = compute_similarity_metric(rs_nodes[0:ventricles,:], ds_nodes[0:ventricles,:])
                        [res_rmse_u, res_max_u, res_sum_u] = compute_similarity_metric(rs_nodes[ventricles:,:], ds_nodes[ventricles:,:])

                        # save results in a different file (stddev, params, similarity)
                        res = '{},{},{},0,1,{},{},{},{},{},{},{},{},{},{},{},{},{},{},{},{},{},{},{},{}\n'.format(case, meta_model, model, stddev[gm_i], stddev[wm_i],
                                                                                                                  len(v_gm), params_gm[0], params_gm[1], params_wm[0], params_wm[1],
                                                                                                                  cost_gm[gm_i], cost_wm[wm_i],
                                                                                                                  rs_filename,ds_filename,
                                                                                                                  rmse_u, max_u, sum_u, ven_rmse_u, ven_max_u, ven_sum_u, res_rmse_u, res_max_u, res_sum_u)

                    res_file.write(res)

                    wm_i += 1
                    experiment += 1

                gm_i += 1

            res_file.close()

            print('stddev[{}]={}'.format(len(stddev),stddev))
            print('value_gm[{}]={}'.format(len(value_gm), value_gm))
            print('cost_gm[{}]={}'.format(len(cost_gm), cost_gm))
            print('value_wm[{}]={}'.format(len(value_wm), value_wm))
            print('cost_wm[{}]={}'.format(len(cost_wm), cost_wm))
            stddev = np.asarray(stddev)[:, None]
            value_gm = np.asarray(value_gm)[:, None]
            cost_gm = np.asarray(cost_gm)[:, None]
            value_wm = np.asarray(value_wm)[:, None]
            cost_wm = np.asarray(cost_wm)[:, None]
            # res = np.hstack([stddev, value_gm, cost_gm, value_wm, cost_wm])
            # print("\n".join(" ".join(map(str, line)) for line in res))