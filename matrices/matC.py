import numpy as np
from geometry.triop import triop

def matC_caliter(psi, phi, Qprev, wprev, tdoa, vprev, d1, P, c):
    """
    Build the C matrix used in orientation (Q, w) updates.
    Rows: [C_Qx; C_Qy; C_Qz; C_wx; C_wy]
    """
    N = P - 1
    C = []
    # Q-related x,y,z
    for idx in range(1, P):
        cQx = (vprev[0] - d1) * triop(3, psi[0] + Qprev, phi[0] + wprev) \
             - (vprev[idx] - d1 - c * tdoa[idx]) * triop(3, psi[idx] + Qprev, phi[idx] + wprev)
        C.append([cQx, 0])
    for idx in range(1, P):
        cQy = (vprev[0] - d1) * triop(4, psi[0] + Qprev, phi[0] + wprev) \
             - (vprev[idx] - d1 - c * tdoa[idx]) * triop(4, psi[idx] + Qprev, phi[idx] + wprev)
        C.append([cQy, 0])
    for idx in range(1, P):
        cQz = (vprev[idx] - d1 - c * tdoa[idx]) * np.sin(psi[idx] + Qprev) \
             - (vprev[0] - d1) * np.sin(psi[0] + Qprev)
        C.append([cQz, 0])
    # w-related x,y
    for idx in range(1, P):
        cwx = -(vprev[0] - d1) * triop(2, psi[0] + Qprev, phi[0] + wprev) \
             + (vprev[idx] - d1 - c * tdoa[idx]) * triop(2, psi[idx] + Qprev, phi[idx] + wprev)
        C.append([0, cwx])
    for idx in range(1, P):
        cwy = (vprev[0] - d1) * triop(1, psi[0] + Qprev, phi[0] + wprev) \
             - (vprev[idx] - d1 - c * tdoa[idx]) * triop(1, psi[idx] + Qprev, phi[idx] + wprev)
        C.append([0, cwy])

    return np.array(C)
