# main.py
import numpy as np
import matplotlib.pyplot as plt
import config
from geometry.angle_distance_setting import angle_dist_setting
from estimation.initial_step     import initial_step_for_1, initial_step_for_2
from estimation.loop_step        import loop_step_for_GAMP_1, loop_step_for_GAMP_2, loop_step_for_inverse_1, loop_step_for_inverse_2
from utils.plotting              import plot_results, plot_coordinates_estimate_compare, plot_error_compare, plot_time_compare, plot_time_error_compare
from utils.printing              import print_results

def main():
    c = config.SPEED_OF_LIGHT
    P = 4
    
    test_1 = 1
    test_2 = 1
    
    # Individual Test
    if test_1==1:
        v_true, d_true, sc, alpha, theta, psi, phi, tdoa, var_tdoa, rho \
            = angle_dist_setting(config.SV, config.HV, config.SCATTERERS,
                                c, config.Q_TRUE, config.W_TRUE)

        # store D1 if you need it (distance of first path)
        config.D1 = d_true[0]
        d1 = config.D1
        
        # GAMP 1
        Q0_1, w0_1, v0_1, var_B_1, elasped_init_1 = initial_step_for_1(alpha, theta, psi, phi, tdoa, var_tdoa, rho, d1, P, c)
        history_1, elasped_loop_1, iterations_1 = loop_step_for_GAMP_1(Q0_1, w0_1, v0_1, var_B_1, config.TOL, alpha, theta, psi, phi, tdoa, var_tdoa, rho, d1, P, c, iterprint=0)
        
        # GAMP 2
        Q0_2, w0_2, v0_2, var_B_2, elasped_init_2 = initial_step_for_2(psi, phi, var_tdoa, rho, d1, P, c)
        history_2, elasped_loop_2, iterations_2 = loop_step_for_GAMP_2(Q0_2, w0_2, v0_2, var_B_2, config.TOL, alpha, theta, psi, phi, tdoa, var_tdoa, rho, d1, P, c, iterprint=0)

        # Inverse 1
        Q0_3, w0_3, v0_3, _, elasped_init_3 = initial_step_for_1(alpha, theta, psi, phi, tdoa, var_tdoa, rho, d1, P, c)
        history_3, elasped_loop_3, iterations_3 = loop_step_for_inverse_1(Q0_3, w0_3, v0_3, config.TOL, alpha, theta, psi, phi, rho, d1, P, c, iterprint=0)

        # Inverse 2
        Q0_4, w0_4, v0_4, _, elasped_init_4 = initial_step_for_2(psi, phi, var_tdoa, rho, d1, P, c)
        history_4, elasped_loop_4, iterations_4 = loop_step_for_inverse_2(Q0_4, w0_4, v0_4, config.TOL, alpha, theta, psi, phi, tdoa, var_tdoa, rho, d1, P, c, iterprint=0)
        
        print(history_1['HV'][-1])
        # Results
        print_results(iterations_1, iterations_2, iterations_3, iterations_4, elasped_init_1, elasped_init_2, elasped_init_3, elasped_init_4, elasped_loop_1, elasped_loop_2, elasped_loop_3, elasped_loop_4)
        '''plot_coordinates_estimate_compare(history_1, history_2, history_3, history_4, config.SV, config.HV, sc, config.Q_TRUE, config.W_TRUE)
        plot_error_compare(history_1, history_2, history_3, history_4)
        plot_time_compare(history_1, history_2, history_3, history_4)
        plot_time_error_compare(history_1, history_2, history_3, history_4)'''
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
        ax.plot(est[:,0], est[:,1], est[:,2], c='g', label='est path')
        ax.plot(est2[:,0], est2[:,1], est2[:,2], c='r', label='est path')
        ax.legend()
        plt.show()
        

    # Mean Test
    elif test_2==1:
        repeat = 100
        HISTORY_1 = {'iter': [], 'time': [], 'accumulated_time': np.zeros((300,repeat)), 'error': np.zeros((300,repeat))}
        HISTORY_2 = {'iter': [], 'time': [], 'accumulated_time': np.zeros((300,repeat)), 'error': np.zeros((300,repeat))}
        HISTORY_3 = {'iter': [], 'time': [], 'accumulated_time': np.zeros((300,repeat)), 'error': np.zeros((300,repeat))}
        HISTORY_4 = {'iter': [], 'time': [], 'accumulated_time': np.zeros((300,repeat)), 'error': np.zeros((300,repeat))}
        
        for i in range(repeat):
            config.SCATTERERS = [
                [10.0, 0.0, 5 + 4*np.random.rand()],
                [17.5, 0.0, 5 + 4*np.random.rand()],
                [17.5, 10.0, 5 + 4*np.random.rand()],
                [10.0, 10.0, 5 + 4*np.random.rand()],
                [20.0, 10.0, 5 + 4*np.random.rand()],
                [20.0, 3.0, 5 + 4*np.random.rand()],
                [15.0, 10.0, 5 + 4*np.random.rand()],
                [12.0, 0.0, 5 + 4*np.random.rand()],
                [24.0, 5.0, 5 + 4*np.random.rand()],
                [12.0, 10.0, 5 + 4*np.random.rand()]
            ]
            #config.Q_TRUE = 15.0 + np.random.rand()
            #config.W_TRUE = 45.0 + np.random.rand()
            
            v_true, d_true, sc, alpha, theta, psi, phi, tdoa, var_tdoa, rho \
            = angle_dist_setting(config.SV, config.HV, config.SCATTERERS,
                                c, config.Q_TRUE, config.W_TRUE)

            # store D1 if you need it (distance of first path)
            config.D1 = d_true[0]
            d1 = config.D1
            
            # GAMP 1
            Q0_1, w0_1, v0_1, var_B_1, elasped_init_1 = initial_step_for_1(alpha, theta, psi, phi, tdoa, var_tdoa, rho, d1, P, c)
            history_1, elasped_loop_1, iterations_1 = loop_step_for_GAMP_1(Q0_1, w0_1, v0_1, var_B_1, config.TOL, alpha, theta, psi, phi, tdoa, var_tdoa, rho, d1, P, c, iterprint=0)
            
            # GAMP 2
            Q0_2, w0_2, v0_2, var_B_2, elasped_init_2 = initial_step_for_2(psi, phi, var_tdoa, rho, d1, P, c)
            history_2, elasped_loop_2, iterations_2 = loop_step_for_GAMP_2(Q0_2, w0_2, v0_2, var_B_2, config.TOL, alpha, theta, psi, phi, tdoa, var_tdoa, rho, d1, P, c, iterprint=0)

            # Inverse 1
            Q0_3, w0_3, v0_3, _, elasped_init_3 = initial_step_for_1(alpha, theta, psi, phi, tdoa, var_tdoa, rho, d1, P, c)
            history_3, elasped_loop_3, iterations_3 = loop_step_for_inverse_1(Q0_3, w0_3, v0_3, config.TOL, alpha, theta, psi, phi, rho, d1, P, c, iterprint=0)

            # Inverse 2
            Q0_4, w0_4, v0_4, _, elasped_init_4 = initial_step_for_2(psi, phi, var_tdoa, rho, d1, P, c)
            history_4, elasped_loop_4, iterations_4 = loop_step_for_inverse_2(Q0_4, w0_4, v0_4, config.TOL, alpha, theta, psi, phi, tdoa, var_tdoa, rho, d1, P, c, iterprint=0)
            
            HISTORY_1['iter'].append(iterations_1)
            HISTORY_2['iter'].append(iterations_2)
            HISTORY_3['iter'].append(iterations_3)
            HISTORY_4['iter'].append(iterations_4)
            HISTORY_1['time'].append(elasped_init_1+elasped_loop_1)
            HISTORY_2['time'].append(elasped_init_2+elasped_loop_2)
            HISTORY_3['time'].append(elasped_init_3+elasped_loop_3)
            HISTORY_4['time'].append(elasped_init_4+elasped_loop_4)
            HISTORY_1['accumulated_time'][:iterations_1, i] = history_1['accumulated_time']
            HISTORY_1['accumulated_time'][iterations_1:, i] = history_1['accumulated_time'][-1]
            HISTORY_2['accumulated_time'][:iterations_2, i] = history_2['accumulated_time']
            HISTORY_2['accumulated_time'][iterations_2:, i] = history_2['accumulated_time'][-1]
            HISTORY_3['accumulated_time'][:iterations_3, i] = history_3['accumulated_time']
            HISTORY_3['accumulated_time'][iterations_3:, i] = history_3['accumulated_time'][-1]
            HISTORY_4['accumulated_time'][:iterations_4, i] = history_4['accumulated_time']
            HISTORY_4['accumulated_time'][iterations_3:, i] = history_4['accumulated_time'][-1]
            HISTORY_1['error'][:iterations_1, i] = history_1['error']
            HISTORY_1['error'][iterations_1:, i] = history_1['error'][-1]
            HISTORY_2['error'][:iterations_2, i] = history_2['error']
            HISTORY_2['error'][iterations_2:, i] = history_2['error'][-1]
            HISTORY_3['error'][:iterations_3, i] = history_3['error']
            HISTORY_3['error'][iterations_3:, i] = history_3['error'][-1]
            HISTORY_4['error'][:iterations_4, i] = history_4['error']
            HISTORY_4['error'][iterations_4:, i] = history_4['error'][-1]
            print(f"Round {i+1:d} Complete!")
        
        HISTORY_1_ACCUMULATED_TIME_MEAN = np.mean(HISTORY_1['accumulated_time'], axis=1, keepdims=True)
        HISTORY_2_ACCUMULATED_TIME_MEAN = np.mean(HISTORY_2['accumulated_time'], axis=1, keepdims=True)
        HISTORY_3_ACCUMULATED_TIME_MEAN = np.mean(HISTORY_3['accumulated_time'], axis=1, keepdims=True)
        HISTORY_4_ACCUMULATED_TIME_MEAN = np.mean(HISTORY_4['accumulated_time'], axis=1, keepdims=True)
        HISTORY_1_ERROR_MEAN = np.mean(HISTORY_1['error'], axis=1, keepdims=True)
        HISTORY_2_ERROR_MEAN = np.mean(HISTORY_2['error'], axis=1, keepdims=True)
        HISTORY_3_ERROR_MEAN = np.mean(HISTORY_3['error'], axis=1, keepdims=True)
        HISTORY_4_ERROR_MEAN = np.mean(HISTORY_4['error'], axis=1, keepdims=True)
        HISTORY_1_ITER_MAX = np.max(HISTORY_1['iter'])
        HISTORY_2_ITER_MAX = np.max(HISTORY_2['iter'])
        HISTORY_3_ITER_MAX = np.max(HISTORY_3['iter'])
        HISTORY_4_ITER_MAX = np.max(HISTORY_4['iter'])
        
        print(f"Average Time Consumed for GAMP1 : {np.mean(HISTORY_1['time'])*1000:.3f} ms")
        print(f"Average Time Consumed for GAMP2 : {np.mean(HISTORY_2['time'])*1000:.3f} ms")
        print(f"Average Time Consumed for Inverse1 : {np.mean(HISTORY_3['time'])*1000:.3f} ms")
        print(f"Average Time Consumed for Inverse2 : {np.mean(HISTORY_4['time'])*1000:.3f} ms")
        print(f"Average Iteration Consumed for GAMP1 : {np.mean(HISTORY_1['iter']):.1f}")
        print(f"Average Iteration Consumed for GAMP2 : {np.mean(HISTORY_2['iter']):.1f}")
        print(f"Average Iteration Consumed for Inverse1 : {np.mean(HISTORY_3['iter']):.1f}")
        print(f"Average Iteration Consumed for Inverse2 : {np.mean(HISTORY_4['iter']):.1f}")

        
        plt.figure()
        plt.plot(HISTORY_1_ACCUMULATED_TIME_MEAN[:HISTORY_1_ITER_MAX], label='GAMP1', linewidth=2)
        plt.plot(HISTORY_2_ACCUMULATED_TIME_MEAN[:HISTORY_2_ITER_MAX], label='GAMP2', linewidth=2)
        plt.plot(HISTORY_3_ACCUMULATED_TIME_MEAN[:HISTORY_3_ITER_MAX], label='Inverse1', linewidth=2)
        plt.plot(HISTORY_4_ACCUMULATED_TIME_MEAN[:HISTORY_4_ITER_MAX], label='Inverse2', linewidth=2)
        plt.xlabel("iterations")
        plt.ylabel("seconds")
        plt.title("Accumulated Time per Iteration")
        plt.legend()
        
        plt.figure()
        plt.plot(HISTORY_1_ERROR_MEAN[:HISTORY_1_ITER_MAX], label='GAMP1', linewidth=2)
        plt.plot(HISTORY_2_ERROR_MEAN[:HISTORY_2_ITER_MAX], label='GAMP2', linewidth=2)
        plt.plot(HISTORY_3_ERROR_MEAN[:HISTORY_3_ITER_MAX], label='Inverse1', linewidth=2)
        plt.plot(HISTORY_4_ERROR_MEAN[:HISTORY_4_ITER_MAX], label='Inverse2', linewidth=2)
        plt.xlabel("iterations")
        plt.ylabel("error [m]")
        plt.hlines(0.3, 0, max(HISTORY_1_ITER_MAX,HISTORY_2_ITER_MAX, HISTORY_3_ITER_MAX, HISTORY_4_ITER_MAX), 'r','--', label='Tolerance')
        plt.title("Error per Iteration")
        plt.legend()
        
        plt.figure()
        plt.scatter(HISTORY_1_ACCUMULATED_TIME_MEAN, HISTORY_1_ERROR_MEAN, alpha=0.8, label='GAMP1')
        plt.scatter(HISTORY_2_ACCUMULATED_TIME_MEAN, HISTORY_2_ERROR_MEAN, alpha=0.8, label='GAMP2')
        plt.scatter(HISTORY_3_ACCUMULATED_TIME_MEAN, HISTORY_3_ERROR_MEAN, alpha=0.8, label='Inverse1')
        plt.scatter(HISTORY_4_ACCUMULATED_TIME_MEAN, HISTORY_4_ERROR_MEAN, alpha=0.8, label='Inverse2')
        plt.hlines(0.3, 0, max(max(HISTORY_1_ACCUMULATED_TIME_MEAN),max(HISTORY_2_ACCUMULATED_TIME_MEAN), max(HISTORY_3_ACCUMULATED_TIME_MEAN), max(HISTORY_4_ACCUMULATED_TIME_MEAN)), 'r','--', label='Tolerance')
        plt.xlabel('seconds')
        plt.ylabel('error [m]')
        plt.title('Accumulated Time Mean - Error Mean')
        plt.legend()
        plt.show()
        
    
if __name__ == '__main__':
    main()