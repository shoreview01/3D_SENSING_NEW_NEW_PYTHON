import numpy as np
import config
from estimation.initial_step import initial_step_for_1, initial_step_for_2
from estimation.loop_step import loop_step_for_GAMP_1, loop_step_for_GAMP_2


def model1(alpha, theta, psi, phi, tdoa, var_tdoa, rho, d1, P, c, iterprint):
    # GAMP 1
    Q0_1, w0_1, v0_1, var_B_1, elasped_init_1 = initial_step_for_1(alpha, theta, psi, phi, tdoa, var_tdoa, rho, d1, P, c)
    history_1, elasped_loop_1, iterations_1 = loop_step_for_GAMP_1(Q0_1, w0_1, v0_1, var_B_1, config.TOL, alpha, theta, psi, phi, tdoa, var_tdoa, rho, d1, P, c, iterprint)
    return history_1, elasped_loop_1, iterations_1

def model2(alpha, theta, psi, phi, tdoa, var_tdoa, rho, d1, P, c, iterprint):
    Q0_2, w0_2, v0_2, var_B_2, elasped_init_2 = initial_step_for_2(psi, phi, var_tdoa, rho, d1, P, c)
    history_2, elasped_loop_2, iterations_2, M, G = loop_step_for_GAMP_2(Q0_2, w0_2, v0_2, var_B_2, config.TOL, alpha, theta, psi, phi, tdoa, var_tdoa, rho, d1, P, c, iterprint)
    return history_2, elasped_loop_2, iterations_2, M, G