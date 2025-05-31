import numpy as np
from geometry.triop import triop

def matB_caliter(psi, phi, Qprev, wprev, rho, d1, P, c):
    """
    Build the B vector for iteration (noisy or true) using TDoA rho.
    """
    N = P - 1
    B = []

    # x-terms
    for idx in range(1, P):
        dcos = c * rho[idx] * triop(1, psi[idx] + Qprev, phi[idx] + wprev)
        acos_line = triop(1, psi[idx] + Qprev, phi[idx] + wprev) - triop(1, psi[0] + Qprev, phi[0] + wprev)
        B.append(-dcos - d1 * acos_line)

    # y-terms
    for idx in range(1, P):
        dsin = -c * rho[idx] * triop(2, psi[idx] + Qprev, phi[idx] + wprev)
        asin_line = triop(2, psi[idx] + Qprev, phi[idx] + wprev) - triop(2, psi[0] + Qprev, phi[0] + wprev)
        B.append(dsin - d1 * asin_line)

    # z-terms
    for idx in range(1, P):
        belev = -c * rho[idx] * np.cos(psi[idx] + Qprev)
        aelev_line = np.cos(psi[idx] + Qprev) - np.cos(psi[0] + Qprev)
        B.append(belev - d1 * aelev_line)

    return np.array(B)


def matB_calvar(psi, phi, Qprev, wprev, P, c, tdoa_var):
    """
    Compute the variance vector for B based on TDoA variance.
    """
    N = P - 1
    Bvar = []

    # x-variances
    for idx in range(1, P):
        var_x = (c**2) * (triop(1, psi[idx] + Qprev, phi[idx] + wprev)**2) * tdoa_var
        Bvar.append(var_x)

    # y-variances
    for idx in range(1, P):
        var_y = (c**2) * (triop(2, psi[idx] + Qprev, phi[idx] + wprev)**2) * tdoa_var
        Bvar.append(var_y)

    # z-variances
    for idx in range(1, P):
        var_z = (c**2) * (np.cos(psi[idx] + Qprev)**2) * tdoa_var
        Bvar.append(var_z)

    return np.array(Bvar)
