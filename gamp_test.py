import numpy as np
import time

A = np.array([[1,0,3],[0,5,0],[0,6,0],[1,0,0],[2,0,1],[0,0,7]])
B = np.array([5,1,9,3,1,2])
x_prev = np.zeros((3,))
M, N = A.shape
x_j = x_prev
var_xj0 = np.ones(N) # prior variance
var_xj = var_xj0
s_i = np.ones(M)
#var_wi = var_B
var_wi = np.ones(M)
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
    var_xj = var_rj * var_xj0 / (var_xj0 + var_rj)
    #print(f"var_xj : {var_xj[-1]:f}")
end1 = time.time()
el1 = end1 - start
print(x_j)

print(np.linalg.pinv(A) @ B)
el2 = time.time() - end1
print(el1*1000)
print(el2*1000)