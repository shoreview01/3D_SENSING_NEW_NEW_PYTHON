import numpy as np
import matplotlib.pyplot as plt
from mpl_toolkits.mplot3d import Axes3D
from matplotlib import cm
import config_SC
from plot_2 import plot_3d
from model import model1, model2
from geometry.angle_distance_setting import angle_dist_setting

def main():
    c = config_SC.SPEED_OF_LIGHT
    P = config_SC.P
    v_true, d_true, sc, alpha, theta, psi, phi, tdoa, var_tdoa, rho \
                = angle_dist_setting(config_SC.SV, config_SC.HV, config_SC.SCATTERERS,
                                    c, config_SC.Q_TRUE, config_SC.W_TRUE)
    # store D1 if you need it (distance of first path)
    config_SC.D_TRUE.append(d_true)
    config_SC.D1 = d_true[0]
    d1 = config_SC.D1
    
    history_1, elasped_loop_1, iterations_1 = model1(alpha, theta, psi, phi, tdoa, var_tdoa, rho, d1, P, c, iterprint=1)
    print("=====================================================")
    history_2, elasped_loop_2, iterations_2, M, G = model2(alpha, theta, psi, phi, tdoa, var_tdoa, rho, d1, P, c, iterprint=1)
    

if __name__ == '__main__':
    main()