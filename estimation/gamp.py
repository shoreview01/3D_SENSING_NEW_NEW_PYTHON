import numpy as np

def gamp(A, B, var_B, x_prev, z_true, P, max_iter=10):
    """
    Generalized AMP for solving A x = B with variances var_B.
    Args:
        A: (M,N) transform matrix
        B: (M,) measurements
        var_B: (M,) variances of B
        x_prev: (N,) initial x
        z_true: (M,) true noiseless Ax for diagnostics
        P: problem size (equals N)
        max_iter: maximum iterations
    Returns:
        x: final estimate of x
    """
    M, N = A.shape
    hist = np.zeros((max_iter,N))
    x_j = x_prev
    var_xj0 = 1e-4 * np.ones(N) # prior variance
    var_xj = var_xj0
    s_i = np.ones(M)
    #var_wi = var_B
    var_wi = 1e-4 * np.ones(M)
    
    for i in range(max_iter):
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
        var_xj = var_rj * var_xj0 / (var_xj0 + var_rj)
        hist[i,:] = x_j
    #print(f"var_xj : {var_xj[-1]:f}")
    return x_j, var_xj, hist