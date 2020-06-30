"""
Plot results related to the Generative Model
"""

import os
import math
import pandas as pd
import numpy as np
from numpy.polynomial.polynomial import polyfit
from scipy import stats
from scipy import special
from itertools import combinations
import matplotlib
from mpl_toolkits.mplot3d import Axes3D
import matplotlib.pyplot as plt
import seaborn as sns

from matplotlib.patches import Circle, PathPatch
import mpl_toolkits.mplot3d.art3d as art3d


def read_data(filename):
    columns = ['Task','Neo-Hookean','Mooney-Rivlin','Ogden1-low','Ogden1-high']

    # data
    data_df = pd.read_csv(filename, sep=",")
    data_df.columns = columns

    return data_df



def plot_metamodel_cost(df):
    dd = pd.melt(df, id_vars=['Task'], value_vars=['Neo-Hookean', 'Mooney-Rivlin', 'Ogden1-low', 'Ogden1-high'], var_name='Metamodels')
    sns.boxplot(x='Task', y='value', data=dd, hue='Metamodels')
    plt.title('Optimisation cost of meta models')
    plt.show()



def stats_metamodel_cost(df):
    tasks = np.unique(df.Task)
    M = ['Neo-Hookean','Mooney-Rivlin','Ogden1-low','Ogden1-high']

    # number of combinations for Bonferroni correction
    #   k       k!
    #  ( ) = ----------
    #   2     2!(k-2)!
    k = len(M)
    comparisons = math.factorial(k) / (math.factorial(2) * math.factorial(k-2))
    p = 0.05
    p_bc = np.around(p / comparisons, decimals=4)

    # iterate through tasks
    for t in tasks:
        df_task = df[df.Task == t]
        print(df_task)

        # ANOVA
        anova_F, anova_p = stats.f_oneway(df_task[M[0]], df_task[M[1]], df_task[M[2]])
        print('ANOVA: F={:.2f} p={:.3f} < p_bc={:.3f}'.format(anova_F, anova_p, p_bc))

        # Kuskal
        kruskal_H, kruskal_p = stats.kruskal(df_task[M[0]], df_task[M[1]], df_task[M[2]])
        print('Kruskal: H={:.2f} p={:.3f} < p_bc={:.3f}'.format(kruskal_H, kruskal_p, p_bc))

        for pairs in combinations([0, 1, 2, 3], 2):
            (mi, mj) = pairs
            print('{} vs {}: '.format(M[mi], M[mj]))
            # print(df_task[M[mi]])
            # print(df_task[M[mj]])

            if all(df_task[M[mi]]-df_task[M[mj]] == 0.0):
                print('   all values are equal')
            else:
                ttest = stats.ttest_rel(df_task[M[mi]], df_task[M[mj]])
                wilcoxon = stats.wilcoxon(df_task[M[mi]], df_task[M[mj]])

                print('   paired t-test = {:.2f} (p={:.3f})'.format(ttest.statistic, ttest.pvalue))
                print('   Wilcoxon test = {:.2f} (p={:.3f})'.format(wilcoxon.statistic, wilcoxon.pvalue))



def plot_boxplot_metamodels_costs():
    """ Data """
    filename = '.\\data\\MetamodelCostNew.csv'
    data = read_data(filename)
    print("Data:")
    print(data)

    hyp_data = data[data['Task']>=9]
    plot_metamodel_cost(hyp_data)

    plot_metamodel_cost(data)
    stats_metamodel_cost(data)


# def plot_series_metamodels_cases():
#     # refer to LeastSquaresLogNH (MICCAI2019) and HyperInference
#     return


def plot_rmse_singletissue(_dir, filename, ref, type='point'):
    anatomy = [0, 1, 2]
    models = [0, 1, 2]
    anatomy_names = ['GM','WM','NH']
    model_names = ['MRE', 'LIN', 'HYP']
    markers = ['o', '^', '*']
    colours = ['black','deepskyblue','mediumorchid',]

    matplotlib.rcParams.update({'font.size': 12})
    fig = plt.figure(figsize=(6,4))

    metamodel = filename[filename.rfind('_')+1:]

    min_a, min_m, min_mu, min_mres = -1, -1, -1, 1000
    ven_min_a, ven_min_m, ven_min_mu, ven_min_mres = -1, -1, -1, 1000
    res_min_a, res_min_m, res_min_mu, res_min_mres = -1, -1, -1, 1000

    for a in anatomy:
        for m in models:
            full_name = _dir + filename + '/SimulationResults_' + filename + '_' + str(a) + '_' + str(m) + '.txt'
            # print('Processing: ',full_name)
            data_df = pd.read_csv(full_name, sep=",", header=0)

            idx_min = data_df['rmse_u'].idxmin()
            ven_idx_min = data_df['ven_rmse_u'].idxmin()
            res_idx_min = data_df['res_rmse_u'].idxmin()
            df = data_df.iloc[idx_min]
            ven_df = data_df.iloc[ven_idx_min]
            res_df = data_df.iloc[res_idx_min]

            ''' MNI '''
            # if df['rmse_u'] < min_mres:
            #     min_a, min_m, min_mu, min_mres = a, m, df['a1'], df['rmse_u']

            ''' Resection '''
            if ven_df['ven_rmse_u'] < ven_min_mres:
                ven_min_a, ven_min_m, ven_min_mu, ven_min_mres = a, m, ven_df['a1'], ven_df['ven_rmse_u']
            if res_df['res_rmse_u'] < res_min_mres:
                res_min_a, res_min_m, res_min_mu, res_min_mres = a, m, res_df['a1'], res_df['res_rmse_u']

            if metamodel == 'nh':
                ''' MNI '''
                # plt.plot(data_df['a1'], data_df['rmse_u'], markers[a], label=model_names[m] + ' ' + anatomy_names[a], linewidth=4, markersize=8, alpha=0.5, color=colours[m])

                ''' Resection '''
                plt.plot(data_df['a1'], data_df['ven_rmse_u'], markers[a], label=model_names[m] + ' ' + anatomy_names[a], linewidth=4, markersize=8, alpha=0.5, color=colours[m])
                plt.plot(data_df['a1'], data_df['res_rmse_u'], markers[a], linewidth=4, markersize=8, alpha=0.5, color=colours[m])

            if metamodel == 'mr':
                plt.plot(2*(data_df['a1'] + data_df['a2']), data_df['rmse_u'], markers[a], label=model_names[m] + ' ' + anatomy_names[a], linewidth=3, markersize=8, alpha=0.5, color=colours[m])


    ''' MNI '''
    print('[ST] Minimum across distributions [MNI]: a={}, m={}, mu={}, mres={}'.format(min_a, min_m, min_mu, min_mres))

    ''' Resection '''
    print('[ST] Minimum across distributions [ventricles]: a={}, m={}, mu={}, mres={}'.format(ven_min_a, ven_min_m, ven_min_mu, ven_min_mres))
    print('[ST] Minimum across distributions [resection]: a={}, m={}, mu={}, mres={}'.format(res_min_a, res_min_m, res_min_mu, res_min_mres))

    if metamodel == 'nh':
        if type == 'point':
            plt.plot([ref[0]], [0.4], marker='o', c='r', markersize=15, alpha=0.2)  # reference
            plt.plot([ref[0]], [0.4], marker='o', c='w', markersize=12, alpha=0.4)  # reference
            plt.plot([ref[0]], [0.4], marker='o', c='r', markersize=9, alpha=0.6)  # reference
            plt.plot([ref[0]], [0.4], marker='o', c='w', markersize=6, alpha=0.8)  # reference
            plt.plot([ref[0]], [0.4], marker='o', c='r', markersize=3, alpha=1.0)  # reference
        elif type == 'line':
            # plt.hlines(ref[0], 0.0, 40000.0, colors='k', linestyles='solid', label='rest')
            plt.hlines(ref[1], 0.0, 40000.0, colors='k', linestyles='dashed', label='rest (ventricles)')
            plt.hlines(ref[2], 0.0, 40000.0, colors='k', linestyles='dotted', label='rest (resection)')
    elif metamodel == 'mr':
        if type == 'point':
            plt.plot([2 * (ref[0] + ref[1])], [0.4], marker='o', c='r', markersize=15, alpha=0.2)  # reference
            plt.plot([2 * (ref[0] + ref[1])], [0.4], marker='o', c='w', markersize=12, alpha=0.4)  # reference
            plt.plot([2 * (ref[0] + ref[1])], [0.4], marker='o', c='r', markersize=9, alpha=0.6)  # reference
            plt.plot([2 * (ref[0] + ref[1])], [0.4], marker='o', c='w', markersize=6, alpha=0.8)  # reference
            plt.plot([2 * (ref[0] + ref[1])], [0.4], marker='o', c='r', markersize=3, alpha=1.0)  # reference
        elif type == 'line':
            plt.hlines(ref[0], 0.0, 40000.0, colors='k', linestyles='--', label='reference')

    plt.xlabel('shear modulus (Pa)')
    plt.ylabel('RMSE (mm)')

    #plt.legend(loc='lower right')  # 'upper center'
    plt.xlim(0, 10000)
    plt.ylim(0, 5)
    # plt.title(filename)
    plt.show()

def plot_rmse_multitissue(title, filename, ref=[], anatomy=0):
    data_df = pd.read_csv(filename, sep=",", header=0)

    var = 'rmse_u'
    if anatomy == 1:
        var = 'ven_rmse_u'
    elif anatomy == 2:
        var = 'res_rmse_u'

    idx_min = data_df[var].idxmin()
    df = data_df.iloc[idx_min]
    # print(df)
    print('[MT] Minimum across distributions [{}]: a={}, m={}, mu={}, sigma={}, mres={}'.format(var, df['anatomy_type_1'], df['method'], df['a1'], df['stddev_1'], df[var]))
    print('[MT] Minimum across distributions [{}]: a={}, m={}, mu={}, sigma={}, mres={}'.format(var, df['anatomy_type_2'], df['method'], df['b1'], df['stddev_2'], df[var]))

    # plot rmse against 2d plot of stddev + a cross where the reference model is [similar to Martinez2013]
    matplotlib.rcParams.update({'font.size': 12})
    fig = plt.figure(figsize=(6,4))
    ax = fig.gca(projection='3d')
    # surf = ax.plot_trisurf(data_df['stddev_1'], data_df['stddev_2'], data_df['rmse_u']*100.0, cmap=plt.cm.viridis, linewidth=0.2)
    surf = ax.plot_trisurf(data_df['stddev_1'], data_df['stddev_2'], data_df[var], cmap='RdPu_r', linewidth=0.2)

    # reference
    if ref != []:
        x_ = np.linspace(-2, 2., 17)
        y_ = np.linspace(-2, 2., 17)
        xx, yy = np.meshgrid(x_, y_)
        zz = np.zeros((17, 17)) + ref[anatomy]
        surf2 = ax.plot_surface(xx, yy, zz, cmap='viridis', linewidth=0.2, alpha=0.3)

    ax.tricontourf(data_df['stddev_1'], data_df['stddev_2'], data_df[var], zdir='z', offset=0, cmap='RdPu_r', alpha=0.9)

    if ref is []:
        ax.scatter([0.603], [1.439], [0.0], marker='*', c='w', s=60)     # reference
        ax.scatter([df.stddev_1], [df.stddev_2], [0.01], marker='o', c='white', s=60)  # estimated
    else:
        ax.scatter([df.stddev_1], [df.stddev_2], [0.01], marker='o', c='black', s=90*2)  # estimated
        ax.scatter([df.stddev_1], [df.stddev_2], [0.01], marker='o', c='white', s=70*2)  # estimated
        ax.scatter([df.stddev_1], [df.stddev_2], [0.01], marker='o', c='black', s=50*2)  # estimated
        ax.scatter([df.stddev_1], [df.stddev_2], [0.01], marker='o', c='white', s=30*2)  # estimated
        ax.scatter([df.stddev_1], [df.stddev_2], [0.01], marker='o', c='black', s=20*2)  # estimated
        ax.scatter([df.stddev_1], [df.stddev_2], [0.01], marker='o', c='white', s=10*2)  # estimated

    # p = Circle((df.stddev_1, df.stddev_2), 0.5)
    # ax.add_patch(p)
    # art3d.pathpatch_2d_to_3d(p, z=0.1, zdir="z")
    ax.set_xlabel('grey matter (std_dev)')
    ax.set_ylabel('white matter (std_dev)')
    ax.set_zlabel('RMSE (mm)')
    ax.set_xlim(-2, 2)
    ax.set_ylim(-2, 2)
    # ax.set_zlim(0, 20)
    # surf.set_clim([0, 20])
    #surf.set_clim([0, 3])
    fig.colorbar(surf, shrink=0.5, aspect=5)
    ax.view_init(20, 45)
    plt.title(title + ' [' + var + ']')
    plt.show()

    return

def plot_rmse_multitissue_together(half_filename, fine_filename):
    half_df = pd.read_csv(half_filename, sep=",", header=0)
    fine_df = pd.read_csv(fine_filename, sep=",", header=0)

    half_idx_min = half_df['rmse_u'].idxmin()
    fine_idx_min = fine_df['rmse_u'].idxmin()
    min_half_df = half_df.iloc[half_idx_min]
    min_fine_df = fine_df.iloc[fine_idx_min]
    print('HALF[min]', min_half_df)
    print('FINE[min]', min_fine_df)

    # plot rmse against 2d plot of stddev + a cross where the reference model is [similar to Martinez2013]
    matplotlib.rcParams.update({'font.size': 12})
    fig = plt.figure()
    ax = fig.gca(projection='3d')
    half_surf = ax.plot_trisurf(half_df['stddev_1'], half_df['stddev_2'], half_df['rmse_u'] * 100.0, cmap='BuPu_r', linewidth=0.2)
    fine_surf = ax.plot_trisurf(fine_df['stddev_1'], fine_df['stddev_2'], fine_df['rmse_u'] * 100.0, cmap='RdPu_r', linewidth=0.2)
    ax.set_xlabel('grey matter (std_dev)')
    ax.set_ylabel('white matter (std_dev)')
    ax.set_zlabel('RMSE (mm)')
    ax.set_xlim(-2, 2)
    ax.set_ylim(-2, 2)
    ax.set_zlim(0, 8)
    half_surf.set_clim([0, 10])
    fine_surf.set_clim([0, 10])
    fig.colorbar(half_surf, shrink=0.5, aspect=5)
    fig.colorbar(fine_surf, shrink=0.5, aspect=5)
    ax.view_init(30, 45)
    plt.show()

    return


def recompute_rmse(rs_filename, ds_filename, scale=1.0):
    rs_nodes, ds_nodes = [], []
    rs_file = open("C:/UCL/PhysicsSimulation/Unity/EpiNav/Assets/Data/Validation/" + rs_filename, "r")
    ds_file = open("C:/UCL/PhysicsSimulation/Unity/EpiNav/Assets/Data/Validation/" + ds_filename, "r")

    lines = rs_file.readlines()
    for l in lines:
        values_str = l.rstrip().split(' ')
        values = [float(i)*scale for i in values_str]
        rs_nodes.append(values)
    rs_file.close()

    lines = ds_file.readlines()
    for l in lines:
        values_str = l.rstrip().split(' ')
        values = [float(i)*scale for i in values_str]
        ds_nodes.append(values)
    ds_file.close()

    rs_nodes = np.asarray(rs_nodes, dtype=np.float)
    ds_nodes = np.asarray(ds_nodes, dtype=np.float)
    ref_nodes = rs_nodes[:, 1:4]
    def_nodes = ds_nodes[:, 1:4]

    N = 0
    if len(ref_nodes) != len(def_nodes):
        print('ERROR compute_similarity_metric() ::. state vectors have different sizes')
    N = len(ref_nodes)

    # compute displacements
    du = ref_nodes - def_nodes
    u = np.linalg.norm(du, axis=1)
    max_u = np.max(u)
    sum_u = np.sum(u)
    rmse = math.sqrt(sum_u / N)

    print('[rmse, max_u, sum_u]', [rmse, max_u, sum_u])

    return [rmse, max_u, sum_u]

def recompute_rmse_anatomy(rs_filename, ds_filename, scale=1.0, ventricles=0):
    rs_nodes, ds_nodes = [], []
    rs_file = open("C:/UCL/PhysicsSimulation/Unity/EpiNav/Assets/Data/Validation/" + rs_filename, "r")
    ds_file = open("C:/UCL/PhysicsSimulation/Unity/EpiNav/Assets/Data/Validation/" + ds_filename, "r")

    lines = rs_file.readlines()
    for l in lines:
        values_str = l.rstrip().split(' ')
        values = [float(i)*scale for i in values_str]
        rs_nodes.append(values)
    rs_file.close()

    lines = ds_file.readlines()
    for l in lines:
        values_str = l.rstrip().split(' ')
        values = [float(i)*scale for i in values_str]
        ds_nodes.append(values)
    ds_file.close()

    rs_nodes = np.asarray(rs_nodes, dtype=np.float)
    ds_nodes = np.asarray(ds_nodes, dtype=np.float)
    ref_nodes = rs_nodes[:, 1:4]
    def_nodes = ds_nodes[:, 1:4]

    N = 0
    if len(ref_nodes) != len(def_nodes):
        print('ERROR compute_similarity_metric() ::. state vectors have different sizes')
    N = len(ref_nodes)

    # compute displacements
    du_ven = ref_nodes[0:ventricles, :] - def_nodes[0:ventricles, :]
    du_res = ref_nodes[ventricles:, :] - def_nodes[ventricles:, :]
    u_ven = np.linalg.norm(du_ven, axis=1)
    u_res = np.linalg.norm(du_res, axis=1)
    max_u_ven = np.max(u_ven)
    max_u_res = np.max(u_res)
    sum_u_ven = np.sum(u_ven)
    sum_u_res = np.sum(u_res)
    rmse_ven = math.sqrt(sum_u_ven / ventricles)
    rmse_res = math.sqrt(sum_u_res / (N - ventricles))

    print('Ventricles[N={}]: [rmse={}, max_u={}, sum_u={}]'.format(ventricles, rmse_ven, max_u_ven, sum_u_ven))
    print('Resection[N={}]: [rmse={}, max_u={}, sum_u={}]'.format(N-ventricles, rmse_res, max_u_res, sum_u_res))

    return [rmse_ven, max_u_ven, sum_u_ven], [rmse_res, max_u_res, sum_u_res]

# def recompute_st_simulation_results(_dir, case, metamodel, method):
#     folder = 'C:/UCL/PhysicsSimulation/Unity/EpiNav/Assets/Data/Validation/'
#     models = [0,1,2]
#     anatomy = 2
#
#     for m in models:
#         res_filename = folder + 'SimulationResults_' + _dir + '_' + str(m) + '.txt'
#         res_file = open(res_filename, "r")
#         header = 'case,metamodel,method,anatomy_type_1,anatomy_type_2,stddev_1,stddev_2,num_params,a1,a2,b1,b2,cost_1,cost_2,rs,ds,rmse_u,max_u,sum_u\n'
#         res_file.write(header)
#
#         rs_filename = _dir + '_ref.txt'
#         std_dev = np.linspace(-2,2,17)
#
#         for s in range(len(std_dev)):
#             ds_filename = _dir + '_' + str(m) + '_' + s + '.txt'
#
#
#         [rmse, max_u, sum_u] = recompute_rmse(rs_filename, ds_filename)



validation_dir = "C:/UCL/PhysicsSimulation/Unity/EpiNav/Assets/Data/Validation/"

# MICCAI Special Issue: comparison of metamodel performance
# plot_boxplot_metamodels_costs()

# MICCAI Special Issue: comparison of metamodels for particular cases
# plot_series_metamodels_cases()


# re-compute a specific case
# recompute_rmse('MNI_half_mt_nh_ref.txt','MNI_half_mt_nh_274.txt')


''' MNI '''
res = 'fine'
# plot RMSE (single-tissue)
plot_rmse_singletissue(validation_dir, 'MNI_'+res+'_st_nh', [333.28], type='point')
# plot_rmse_singletissue(validation_dir, 'MNI_fine_st_mr', [280,333], type='point')

# plot RMSE (multi-tissue)
filename = validation_dir + 'MNI_'+res+'_mt_nh/SimulationResults_MNI_'+res+'_mt_nh_0_2.txt'
# half_nh_filename = "C:/UCL/PhysicsSimulation/Unity/EpiNav/Assets/Data/Validation/MNI_half_mt_nh/SimulationResults_MNI_half_mt_nh_0_2.txt"
# fine_nh_filename = "C:/UCL/PhysicsSimulation/Unity/EpiNav/Assets/Data/Validation/MNI_fine_mt_nh/SimulationResults_MNI_fine_mt_nh_0_2.txt"
# half_mr_filename = "C:/UCL/PhysicsSimulation/Unity/EpiNav/Assets/Data/Validation/MNI_half_mt_mr/SimulationResults_MNI_half_mt_mr.txt"
# fine_mr_filename = "C:/UCL/PhysicsSimulation/Unity/EpiNav/Assets/Data/Validation/MNI_fine_mt_mr/SimulationResults_MNI_fine_mt_mr.txt"

# plot_rmse_multitissue(res, filename)

# plot_rmse_multitissue_together(half_filename, fine_filename)


''' Resection cases '''
# [rmse, max_u, sum_u]
# case, n_ventricles = '0529', 263
# case, n_ventricles = '0535', 174
# case, n_ventricles = '0555', 140
# case, n_ventricles = '0603', 169
case, n_ventricles = '0614', 196
# case, n_ventricles = '0660', 198
# case, n_ventricles = '0684', 137
# case, n_ventricles = '0685', 164

sim = recompute_rmse(case+'_half_st_nh/'+case+'_half_st_nh_ref.txt', case+'_half_st_nh/'+case+'_half_st_nh_rest.txt', scale=100.0)
sim_ven, sim_res = recompute_rmse_anatomy(case+'_half_st_nh/'+case+'_half_st_nh_ref.txt', case+'_half_st_nh/'+case+'_half_st_nh_rest.txt', scale=100.0, ventricles=n_ventricles)
# print('ST => rest vs reference: [rmse={}, max_u={}, sum_u={}]'.format(sim[0], sim[1], sim[2]))
# print('             ventricles: [rmse={}, max_u={}, sum_u={}]'.format(sim_ven[0], sim_ven[1], sim_ven[2]))
# print('              resection: [rmse={}, max_u={}, sum_u={}]'.format(sim_res[0], sim_res[1], sim_res[2]))
plot_rmse_singletissue(validation_dir, case+'_half_st_nh', [sim[0], sim_ven[0], sim_res[0]], type='line')


res_nh_filename = "C:/UCL/PhysicsSimulation/Unity/EpiNav/Assets/Data/Validation/{}_half_mt_nh/SimulationResults_{}_half_mt_nh_0_2.txt".format(case, case)

plot_rmse_multitissue(case, res_nh_filename, [sim[0], sim_ven[0], sim_res[0]], anatomy=1)
plot_rmse_multitissue(case, res_nh_filename, [sim[0], sim_ven[0], sim_res[0]], anatomy=2)

