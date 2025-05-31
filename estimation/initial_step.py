import numpy as np
import time
import config
from matrices.matA import matA_caliter
from matrices.matB import matB_caliter, matB_calvar
from estimation.gamp import gamp

def initial_step_for_1(alpha, theta, psi, phi, tdoa, var_tdoa, rho, d1, P, c):
    
    start = time.time()
    
    # initial guesses: mean of AoD/AoA angles
    Q0 = np.mean(psi)
    w0 = np.mean(phi)

    # build A and true B
    A0 = matA_caliter(alpha, theta, psi, phi, Q0, w0, P)
    z0 = matB_caliter(psi, phi, Q0, w0, tdoa, d1, P, c)

    # noisy measurements
    B0 = matB_caliter(psi, phi, Q0, w0, rho, d1, P, c)
    var_B = matB_calvar(psi, phi, Q0, w0, P, c, var_tdoa)

    # initial v estimate via GAMP
    v = np.ones(P)
    v0, _, gamp_hist = gamp(A0, B0, var_B, v, z0, P)

    elasped = time.time() - start

    return Q0, w0, v0, var_B, elasped

def initial_step_for_2(psi, phi, var_tdoa, rho, d1, P, c):
    
    start = time.time()
    # initial guesses: mean of AoD/AoA angles
    Q0 = np.mean(psi)
    w0 = np.mean(phi)

    # noisy measurements
    var_B = matB_calvar(psi, phi, Q0, w0, P, c, var_tdoa)

    # initial v guess
    v0 = (rho + d1)/2

    elasped = time.time() - start
    
    return Q0, w0, v0, var_B, elasped