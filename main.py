# main.py
import numpy as np
import matplotlib.pyplot as plt
from mpl_toolkits.mplot3d import Axes3D
from matplotlib import cm
import config
from plot_2 import plot_3d
from model import model1, model2
from geometry.angle_distance_setting import angle_dist_setting
from estimation.initial_step     import initial_step_for_1, initial_step_for_2
from estimation.loop_step        import loop_step_for_GAMP_1, loop_step_for_GAMP_2, loop_step_for_inverse_1, loop_step_for_inverse_2
from utils.plotting              import plot_results, plot_coordinates_estimate_compare, plot_error_compare, plot_time_compare, plot_time_error_compare
from utils.printing              import print_results

def main():
    c = config.SPEED_OF_LIGHT
    P = config.P

    error = []
    minx = 5
    maxx = 25
    miny = -30
    maxy = -5
    # GAMP 1
    for x in range(minx,maxx+1,1):
        for y in range(maxy,miny-1,-1):
            config.HV = [x,y,2.0]
            v_true, d_true, sc, alpha, theta, psi, phi, tdoa, var_tdoa, rho \
                = angle_dist_setting(config.SV, config.HV, config.SCATTERERS,
                                    c, config.Q_TRUE, config.W_TRUE)

            # store D1 if you need it (distance of first path)
            config.D_TRUE.append(d_true)
            config.D1 = d_true[0]
            d1 = config.D1

            history_1, elasped_loop_1, iterations_1 = model1(alpha, theta, psi, phi, tdoa, var_tdoa, rho, d1, P, c, iterprint=0)
            error.append([x, y, np.linalg.norm(config.HV-history_1['HV'][-1])])
    print("====================================================")

    # GAMP 2
    #history_2, elasped_loop_2, iterations_2 = model2(alpha, theta, psi, phi, tdoa, var_tdoa, rho, d1, P, c, iterprint=1)
    plt.plot([err[2] for err in error])
    plot_3d(sc, history_1)
    
    #plot_3d(sc, history_2)
    # Sample data
    x = np.array([a for a in range(minx,maxx+1,1)])
    y = np.array([b for b in range(maxy,miny-1,-1)])
    xpos, ypos = np.meshgrid(x, y, indexing="ij")
    xpos = xpos.flatten()
    ypos = ypos.flatten()
    zpos = np.zeros_like(xpos)

    dx = dy = 0.5
    dz = np.array([row[2] for row in error])

    norm = plt.Normalize(dz.min(), dz.max())
    colors = cm.summer(norm(dz))  # or use 'plasma', 'inferno', etc.


    # Plotting
    fig = plt.figure()
    ax = fig.add_subplot(111, projection='3d')
    ax.bar3d(xpos, ypos, zpos, dx, dy, dz, color=colors)

    ax.set_xlabel('X')
    ax.set_ylabel('Y')
    ax.set_zlabel('Error (m)')
    ax.set_title("Error per HV Position")
    plt.show()
if __name__ == '__main__':
    main()