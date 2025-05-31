# main.py
import numpy as np
import matplotlib.pyplot as plt
import config
from model import model1
from geometry.angle_distance_setting import angle_dist_setting
from estimation.initial_step     import initial_step_for_1, initial_step_for_2
from estimation.loop_step        import loop_step_for_GAMP_1, loop_step_for_GAMP_2, loop_step_for_inverse_1, loop_step_for_inverse_2
from utils.plotting              import plot_results, plot_coordinates_estimate_compare, plot_error_compare, plot_time_compare, plot_time_error_compare
from utils.printing              import print_results

def main():
    c = config.SPEED_OF_LIGHT
    P = config.P

    v_true, d_true, sc, alpha, theta, psi, phi, tdoa, var_tdoa, rho \
        = angle_dist_setting(config.SV, config.HV, config.SCATTERERS,
                            c, config.Q_TRUE, config.W_TRUE)

    # store D1 if you need it (distance of first path)
    config.D_TRUE.append(d_true)
    config.D1 = d_true[0]
    d1 = config.D1
    
    # GAMP 1
    history_1, elasped_loop_1, iterations_1 = model1(alpha, theta, psi, phi, tdoa, var_tdoa, rho, d1, P, c, iterprint=1)
    print(np.cos(psi+config.Q_TRUE))
    print(np.sin(psi+config.Q_TRUE))
    print(np.cos(phi+config.W_TRUE))
    print(np.sin(phi+config.W_TRUE))
    '''# GAMP 2
    history_1, elasped_loop_1, iterations_1 = model2(alpha, theta, psi, phi, tdoa, var_tdoa, rho, d1, P, c, iterprint=0)'''
    
    # Results        
    fig = plt.figure(figsize=(7,7))
    # — 3D scene —
    ax = fig.add_subplot(projection='3d')
    lim = np.max(np.abs(np.vstack([config.SV, config.HV, sc]))) * 1.2
    xs = np.linspace(-10, lim, 2)
    ys = np.linspace(-lim, lim, 2)
    Xs, Ys = np.meshgrid(xs, ys)
    Zs = np.zeros_like(Xs)
    ax.plot_surface(
        Xs, Ys, Zs,
        color='gray', alpha=0.3,
        edgecolor='none', shade=True
    )
    ax.scatter(*config.SV, c='r', label='SV')
    ax.scatter(*config.HV, c='k', label='true HV')
    ax.scatter(*history_1['HV'][-1], c='r', label='est HV')
    for s in sc:
        ax.scatter(*s, c='b', marker='x')
    est = np.array(history_1['all_HV'])
    est2 = np.array(history_1['HV'])
    ax.plot(est2[:,0], est2[:,1], est2[:,2], c='r', label='est path')
    ax.legend()
    plt.show()
        
    
if __name__ == '__main__':
    main()