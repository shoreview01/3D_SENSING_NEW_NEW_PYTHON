import numpy as np
from geometry.triop import triop

def matD_caliter(C, alpha, theta, psi, phi, Qprev, wprev, rho, vprev, d1, P, c):
    """
    Build the D vector used in orientation (Q, w) updates.
    """
    N = P - 1
    D = []
    # Q x,y,z residuals
    for idx in range(1, P):
        delta_x = Qprev * C[idx-1,0] \
                  - (vprev[0] - d1) * triop(1, psi[0]+Qprev, phi[0]+wprev) \
                  + (vprev[idx]-d1-c*rho[idx])*triop(1, psi[idx]+Qprev, phi[idx]+wprev) \
                  - vprev[0] * triop(1, alpha[0], theta[0]) \
                  + vprev[idx] * triop(1, alpha[idx], theta[idx])
        D.append(delta_x)
    for idx in range(1, P):
        delta_y = Qprev * C[N+idx-1,0] \
                  - (vprev[0] - d1) * triop(2, psi[0] + Qprev, phi[0] + wprev) \
                  + (vprev[idx]-d1-c*rho[idx])*triop(2, psi[idx]+Qprev, phi[idx]+wprev) \
                  - vprev[0] * triop(2, alpha[0], theta[0]) \
                  + vprev[idx] * triop(2, alpha[idx], theta[idx])
        D.append(delta_y)
    for idx in range(1, P):
        delta_z = Qprev * C[2*N+idx-1,0] \
                  - vprev[0] * np.cos(alpha[0]) \
                  + vprev[idx] * np.cos(alpha[idx]) \
                  - ((vprev[0]-d1)*np.cos(psi[0]+Qprev) \
                     - (vprev[idx]-d1-c*rho[idx])*np.cos(psi[idx]+Qprev))
        D.append(delta_z)
    # w x,y residuals
    for idx in range(1, P):
        delta_wx = wprev * C[3*N+idx-1,1] \
                   - vprev[0]*triop(1, alpha[0], theta[0]) \
                   + vprev[idx]*triop(1, alpha[idx], theta[idx]) \
                   - ((vprev[0]-d1)*triop(1, psi[0]+Qprev, phi[0]+wprev) \
                      - (vprev[idx]-d1-c*rho[idx])*triop(1, psi[idx]+Qprev, phi[idx]+wprev))
        D.append(delta_wx)
    for idx in range(1, P):
        delta_wy = wprev * C[4*N+idx-1,1] \
                   - vprev[0]*triop(2, alpha[0], theta[0]) \
                   + vprev[idx]*triop(2, alpha[idx], theta[idx]) \
                   - ((vprev[0]-d1)*triop(2, psi[0]+Qprev, phi[0]+wprev) \
                      - (vprev[idx]-d1-c*rho[idx])*triop(2, psi[idx]+Qprev, phi[idx]+wprev))
        D.append(delta_wy)

    return np.array(D)