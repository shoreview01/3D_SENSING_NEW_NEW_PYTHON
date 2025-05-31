# utils/plotting.py

import matplotlib.pyplot as plt
import numpy as np

'''plot_results(config.SV, config.HV, sc, history_1, iterations_1, config.Q_TRUE, config.W_TRUE, fig_id=1)
    plot_results(config.SV, config.HV, sc, history_2, iterations_2, config.Q_TRUE, config.W_TRUE, fig_id=2)
    plot_results(config.SV, config.HV, sc, history_3, iterations_3, config.Q_TRUE, config.W_TRUE, fig_id=3)
    plot_results(config.SV, config.HV, sc, history_4, iterations_4, config.Q_TRUE, config.W_TRUE, fig_id=4)'''

def plot_results(SV, HV, sc, history, iterations, Q_true, w_true, fig_id=None):
    """
    SV, HV      : 3-vectors of true sensor & vehicle
    sc          : list of 3-vectors (scatterers)
    history     : dict with keys 'Q','w','HV' holding per-iter lists
    times       : list of cumulative times
    Q_true,w_true : true orientation scalars (radians)
    """
    iterlist = range(1, iterations+1)
    # create or reuse a figure window
    fig = plt.figure(fig_id, figsize=(15,8))
    fig.clf()  # clear previous contents if reusing
    fig.suptitle(f"Figure {fig_id:d}")
    
    # — 3D scene —
    ax = fig.add_subplot(121, projection='3d')
    lim = np.max(np.abs(np.vstack([SV, HV, sc]))) * 1.2
    xs = np.linspace(-10, lim, 2)
    ys = np.linspace(-lim, lim, 2)
    Xs, Ys = np.meshgrid(xs, ys)
    Zs = np.zeros_like(Xs)
    ax.plot_surface(
        Xs, Ys, Zs,
        color='gray', alpha=0.3,
        edgecolor='none', shade=True
    )
    ax.scatter(*SV, c='r', label='SV')
    ax.scatter(*HV, c='k', label='true HV')
    ax.scatter(*history['HV'][-1], c='r', label='est HV')
    for s in sc:
        ax.scatter(*s, c='b', marker='x')
    est = np.array(history['HV'])
    ax.plot(est[:,0], est[:,1], est[:,2], c='g', label='est path')
    ax.legend()

    # — convergence traces —
    # Q
    ax2 = fig.add_subplot(222)
    ax2.plot(iterlist, np.rad2deg(history['Q']), '-', label='Q')
    ax2.hlines(np.rad2deg(Q_true), iterlist[0], iterlist[-1], 'r','--', label='true Q')
    ax2.set_ylabel('Q (deg)')
    ax2.legend()

    # ω
    ax3 = fig.add_subplot(224)
    ax3.plot(iterlist, np.rad2deg(history['w']), '-', label='ω')
    ax3.hlines(np.rad2deg(w_true), iterlist[0], iterlist[-1], 'r','--', label='true ω')
    ax3.set_ylabel('ω (deg)')
    ax3.set_xlabel('Iterations')
    ax3.legend()

    plt.tight_layout()
    
def plot_coordinates_estimate_compare(history_1, history_2, history_3, history_4, SV, HV, sc, Q_true, w_true, fig_id=None):
    # create or reuse a figure window
    fig = plt.figure(fig_id, figsize=(6,6))
    lim = np.max(np.abs(np.vstack([SV, HV, sc]))) * 1.2
    xs = np.linspace(-10, lim, 2)
    ys = np.linspace(-lim, lim, 2)
    Xs, Ys = np.meshgrid(xs, ys)
    Zs = np.zeros_like(Xs)
    ax = fig.add_subplot(projection='3d')
    ax.plot_surface(
        Xs, Ys, Zs,
        color='gray', alpha=0.3,
        edgecolor='none', shade=True
    )
    ax.scatter(*SV, c='r', label='SV')
    ax.scatter(*HV, c='k', label='true HV')
    ax.scatter(*history_1['HV'][-1], label='est HV1')
    ax.scatter(*history_2['HV'][-1], label='est HV2')
    ax.scatter(*history_3['HV'][-1], label='est HV3')
    ax.scatter(*history_4['HV'][-1], label='est HV4')
    for s in sc:
        ax.scatter(*s, c='b', marker='x')
    est_1 = np.array(history_1['HV'])
    est_2 = np.array(history_2['HV'])
    est_3 = np.array(history_3['HV'])
    est_4 = np.array(history_4['HV'])
    ax.plot(est_1[:,0], est_1[:,1], est_1[:,2], label='GAMP1 path')
    ax.plot(est_2[:,0], est_2[:,1], est_2[:,2], label='GAMP2 path')
    ax.plot(est_3[:,0], est_3[:,1], est_3[:,2], label='Inverse1 path')
    ax.plot(est_4[:,0], est_4[:,1], est_4[:,2], label='Inverse2 path')
    ax.legend()
    

def plot_error_compare(history_1, history_2, history_3, history_4):
    plt.figure(figsize=(8,5))
    plt.plot(range(1,len(history_1['Q'])+1), history_1['error'], label='GAMP1', linewidth=2)
    plt.plot(range(1,len(history_2['Q'])+1), history_2['error'], label='GAMP2', linewidth=2)
    plt.plot(range(1,len(history_3['Q'])+1), history_3['error'], label='Inverse1', linewidth=2)
    plt.plot(range(1,len(history_4['Q'])+1), history_4['error'], label='Inverse2', linewidth=2)
    plt.ylabel("Error")
    plt.xlabel("Iterations")
    plt.title("Coordinates Error per Iteration")
    plt.legend()

def plot_time_compare(history_1, history_2, history_3, history_4):
    plt.figure(figsize=(8,5))
    plt.plot(range(1,len(history_1['Q'])+1), [t * 1000 for t in history_1['accumulated_time']], label='GAMP1', linewidth=2)
    plt.plot(range(1,len(history_2['Q'])+1), [t * 1000 for t in history_2['accumulated_time']], label='GAMP2', linewidth=2)
    plt.plot(range(1,len(history_3['Q'])+1), [t * 1000 for t in history_3['accumulated_time']], label='Inverse1', linewidth=2)
    plt.plot(range(1,len(history_4['Q'])+1), [t * 1000 for t in history_4['accumulated_time']], label='Inverse2', linewidth=2)
    plt.ylabel("milliseconds")
    plt.xlabel("Iterations")
    plt.title("Accumulated Time per Iteration")
    plt.legend()
    
def plot_time_error_compare(history_1, history_2, history_3, history_4):
    plt.figure()
    plt.scatter(history_1['accumulated_time'], history_1['error'], alpha=0.8, label='GAMP1')
    plt.scatter(history_2['accumulated_time'], history_2['error'], alpha=0.8, label='GAMP2')
    plt.scatter(history_3['accumulated_time'], history_3['error'], alpha=0.8, label='Inverse1')
    plt.scatter(history_4['accumulated_time'], history_4['error'], alpha=0.8, label='Inverse2')
    plt.hlines(0.3, 0, max(max(history_1['accumulated_time']),max(history_2['accumulated_time']), max(history_3['accumulated_time']), max(history_4['accumulated_time'])), 'r','--', label='Tolerance')
    plt.xlabel('seconds')
    plt.ylabel('error [m]')
    plt.title('Accumulated Time Mean - Error Mean')
    plt.legend()
