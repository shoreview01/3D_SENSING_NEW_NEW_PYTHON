import numpy as np
import time
import matplotlib.pyplot as plt
from mpl_toolkits.mplot3d import Axes3D
from matplotlib import cm
import config
from plot_2 import plot_3d
from model import model1, model2
from geometry.angle_distance_setting import angle_dist_setting

c = config.SPEED_OF_LIGHT
P = config.P
v_true, d_true, sc, alpha, theta, psi, phi, tdoa, var_tdoa, rho \
            = angle_dist_setting(config.SV, config.HV, config.SCATTERERS,
                                c, config.Q_TRUE, config.W_TRUE)
# store D1 if you need it (distance of first path)
config.D_TRUE.append(d_true)
config.D1 = d_true[0]
d1 = config.D1
history_2, elasped_loop_2, iterations_2, M, G = model2(alpha, theta, psi, phi, tdoa, var_tdoa, rho, d1, P, c, iterprint=1)
A = M
B = G
print(A)
print(B)
x_true = np.linalg.pinv(A)@B
x_prev = np.zeros((6,))
error = []



M, N = A.shape
x_j = x_prev
var_xj0 = 1e-4*np.ones(N) # prior variance
var_xj = var_xj0
s_i = np.ones(M)
#var_wi = var_B
var_wi = 1e-4*np.ones(M)
start = time.time()
for _ in range(20):
    # output linear step
    var_pi = (A**2).dot(var_xj)
    p_i = A.dot(x_j) - var_pi * s_i
    z_i = A.dot(x_j)
    # output nonlinear step
    s_i = (B - p_i) / (var_wi + var_pi)
    var_si = 1.0 / (var_wi + var_pi)
    # input linear step
    var_rj = 1.0 / ((A**2).T.dot(var_si))
    r_j = x_j + var_rj * (A.T.dot(s_i))
    # input nonlinear step (Gaussian prior)
    x_j = r_j
    error.append(np.linalg.norm(x_j-x_true)/np.linalg.norm(x_true)*100)
    var_xj = var_rj * var_xj0 / (var_xj0 + var_rj)
    #print(f"var_xj : {var_xj[-1]:f}")
end1 = time.time()
el1 = end1 - start
print(x_j)
print(x_true)
print(x_j-x_true)
plt.plot([x for x in range(1,21,1)],error, c='navy', linewidth=2)
plt.hlines(1, 1, 20, 'r','--', label='1%')
plt.xlabel("Iteration")
plt.ylabel("Error %")
plt.title("Error by GAMP Iteration")
plt.show()