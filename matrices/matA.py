import numpy as np
from geometry.triop import triop

def matA_caliter(alpha, theta, psi, phi, Qprev, wprev, P):
    """
    Build the matrix A for iteration given orientations Qprev, wprev.
    """
    N = P - 1
    A = np.zeros((3 * N, P))

    # first column (r1 terms)
    for n in range(N):
        ax = triop(1, alpha[0], theta[0]) + triop(1, psi[0] + Qprev, phi[0] + wprev)
        ay = triop(2, alpha[0], theta[0]) + triop(2, psi[0] + Qprev, phi[0] + wprev)
        az = np.cos(alpha[0]) + np.cos(psi[0] + Qprev)
        A[n, 0] = ax
        A[N + n, 0] = ay
        A[2 * N + n, 0] = az

    # remaining columns
    for idx in range(1, P):
        axp = triop(1, alpha[idx], theta[idx]) + triop(1, psi[idx] + Qprev, phi[idx] + wprev)
        ayp = triop(2, alpha[idx], theta[idx]) + triop(2, psi[idx] + Qprev, phi[idx] + wprev)
        azp = np.cos(alpha[idx]) + np.cos(psi[idx] + Qprev)

        A[idx - 1, idx] = -axp
        A[N + idx - 1, idx] = -ayp
        A[2 * N + idx - 1, idx] = -azp

    return A
