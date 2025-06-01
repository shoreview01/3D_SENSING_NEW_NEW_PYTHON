# estimation/loop_step.py
import numpy as np
import time
from matrices.matA import matA_caliter
from matrices.matB import matB_caliter, matB_calvar
from matrices.matC import matC_caliter
from matrices.matD import matD_caliter
from estimation.gamp import gamp
from config import MAX_ITER, TOL, HV, D_TRUE, Q_TRUE, W_TRUE

max_loop_iter = 299

def loop_step_for_GAMP_1(Q0, w0, v0, var_B, TOL, alpha, theta, psi, phi, tdoa, var_tdoa, rho, d1, P, c, iterprint):
    Q_prev, w_prev, v_prev = Q0, w0, v0
    history = {'Q':[], 'w':[], 'HV':[], 'iter':[], 'accumulated_time': [], 'error': [], 'all_HV': []}
    iterations = 0
    deltaQ, deltaw = 1.0, 1.0
    deltav = np.ones(P+1)
    HVh = np.array([0,0,0])
    
    start = time.time()
    while abs(np.rad2deg(deltaQ)) > TOL and abs(np.rad2deg(deltaw)) > TOL and max(abs(deltav) > TOL) and iterations<=max_loop_iter:

        iterations += 1

        # — build C, D, update Q & w —
        C = matC_caliter(psi, phi, Q_prev, w_prev, rho, v_prev, d1, P, c)
        var_C = 1e-4 * np.ones(3*(P-1))
        D = matD_caliter(C, alpha, theta, psi, phi,
                Q_prev, w_prev, rho, v_prev, d1, P, c)
        Qw_prev = np.array([Q_prev, w_prev])
        Qw, _, gamp_hist_qw = gamp(C, D, var_B, Qw_prev, np.zeros((2,1)), P)
        Qn = Qw[0]
        wn = Qw[1]

        # — build A, B and update v via GAMP —
        A      = matA_caliter(alpha, theta, psi, phi, Qn, wn, P)
        z = matB_caliter(psi, phi, Qn, wn, tdoa,  d1, P, c)
        B  = matB_caliter(psi, phi, Qn, wn, rho,   d1, P, c)
        var_B   = matB_calvar(psi, phi, Qn, wn, P, c, var_tdoa)
        vn, var_v, gamp_hist_v     = gamp(A, B, var_B, v_prev, z, P)
        
        # — HV estimate —
        xh, yh, zh = 0,0,0
        for p in range(P):
            xh += (vn[p]*np.sin(alpha[p])*np.cos(theta[p]) \
            - (d1+c*rho[p] - vn[p])*np.sin(psi[p]+Qn)*np.cos(phi[p]+wn))/P
            yh += (vn[p]*np.sin(alpha[p])*np.sin(theta[p]) \
            - (d1+c*rho[p] - vn[p])*np.sin(psi[p]+Qn)*np.sin(phi[p]+wn))/P
            zh += (vn[p]*np.cos(alpha[p]) \
            - (d1+c*rho[p] - vn[p])*np.cos(psi[p]+Qn))/P
        HVh = np.array([xh,yh,zh])

        # — logging —
        history['Q'].append(Qn)
        history['w'].append(wn)
        history['HV'].append(HVh)
        history['error'].append(np.linalg.norm(HV-HVh))

        if iterprint==1:
            print(f"Iter {iterations:2d} | Q={np.rad2deg(Qn):6.3f}°, ω={np.rad2deg(wn):6.3f}°"
                f" | v=({vn[0]:6.2f},{vn[1]:6.2f},{vn[2]:6.2f},{vn[3]:6.2f})"
                f" | HV=({xh:6.2f},{yh:6.2f},{zh:6.2f})")

        deltaQ = Qn - Q_prev
        deltaw = wn - w_prev
        deltav = vn - v_prev
        Q_prev = Qn
        w_prev = wn
        v_prev = vn
        
        iter_time = time.time() - start
        history['accumulated_time'].append(iter_time)
        
    history['iter'].append(iterations)
    end = time.time()
    elasped_gamp_1 = end - start
    iterations = len(history['Q'])
    
    return history, elasped_gamp_1, iterations

def loop_step_for_GAMP_2(Q0, w0, v0, var_B, TOL, alpha, theta, psi, phi, tdoa, var_tdoa, rho, d1, P, c, iterprint):
    Q_prev, w_prev, v_prev = Q0, w0, v0
    history = {'Q':[], 'w':[], 'HV':[], 'iter':[], 'accumulated_time': [], 'error' : []}
    iterations = 0
    deltaQ, deltaw = 1.0, 1.0
    deltav = np.ones(P+1)

    start = time.time()
    while abs(np.rad2deg(deltaQ))>TOL and abs(np.rad2deg(deltaw)) > TOL and max(abs(deltav) > TOL) and iterations<=max_loop_iter :

        iterations += 1

        # — build C, D—
        C = matC_caliter(psi, phi, Q_prev, w_prev, rho, v_prev, d1, P, c)
        var_C = 1e-4 * np.ones(3*(P-1))
        D = matD_caliter(C, alpha, theta, psi, phi,
                Q_prev, w_prev, rho, v_prev, d1, P, c)
        var_D = 1e-4 * np.ones(2*(P-1))
        
        # — build A, B and update Q, w, v via GAMP —
        A      = matA_caliter(alpha, theta, psi, phi, Q_prev, w_prev, P)
        z = matB_caliter(psi, phi, Q_prev, w_prev, tdoa,  d1, P, c)
        B  = matB_caliter(psi, phi, Q_prev, w_prev, rho,   d1, P, c)
        var_B   = matB_calvar(psi, phi, Q_prev, w_prev, P, c, var_tdoa)
        M = np.zeros((8*(P-1),2+P))
        M[:3*(P-1), 2:]   = A    # top-right
        M[3*(P-1):,  :2]  = C    # bottom-left
        G = np.zeros((8*(P-1),))
        G[:3*(P-1)]   = B   # top-right
        G[3*(P-1):]  = D   # bottom-left
        xn, var_xn, gamp_hist     = gamp(M, G, var_B, np.concatenate([np.array([Q_prev, w_prev]), v_prev]).reshape(2+P,), z, P)
        Qn = xn[0]
        wn = xn[1]
        vn = xn[2:]
        
        # — HV estimate —
        xh, yh, zh = 0,0,0
        for p in range(P):
            xh += (vn[p]*np.sin(alpha[p])*np.cos(theta[p]) \
            - (d1+c*rho[p] - vn[p])*np.sin(psi[p]+Qn)*np.cos(phi[p]+wn))/P
            yh += (vn[p]*np.sin(alpha[p])*np.sin(theta[p]) \
            - (d1+c*rho[p] - vn[p])*np.sin(psi[p]+Qn)*np.sin(phi[p]+wn))/P
            zh += (vn[p]*np.cos(alpha[p]) \
            - (d1+c*rho[p] - vn[p])*np.cos(psi[p]+Qn))/P
        HVh = np.array([xh,yh,zh])

        # — logging —
        history['Q'].append(Qn)
        history['w'].append(wn)
        history['HV'].append(HVh)
        history['error'].append(np.linalg.norm(HV-HVh))

        if iterprint==1:
            print(f"Iter {iterations:2d} | Q={np.rad2deg(Qn):6.3f}°, ω={np.rad2deg(wn):6.3f}°"
                f" | HV=({xh:6.2f},{yh:6.2f},{zh:6.2f})")
            

        deltaQ = Qn - Q_prev
        deltaw = wn - w_prev
        deltav = vn - v_prev
        Q_prev = Qn
        w_prev = wn
        v_prev = vn
        iter_time = time.time() - start
        history['accumulated_time'].append(iter_time)
        
    history['iter'].append(iterations)
    end = time.time()
    elasped_gamp_2 = end - start
    iterations = len(history['Q'])
    
    return history, elasped_gamp_2, iterations

def loop_step_for_inverse_1(Q0, w0, v0, TOL, alpha, theta, psi, phi, rho, d1, P, c, iterprint):
    Q_prev, w_prev, v_prev = Q0, w0, v0
    history = {'Q':[], 'w':[], 'HV':[], 'iter':[], 'accumulated_time' : [], 'error' : []}
    iterations = 0
    deltaQ, deltaw = 1.0, 1.0
    deltav = np.ones(P+1)
    HVh = np.array([0,0,0])
    start = time.time()
    while abs(np.rad2deg(deltaQ))>TOL and abs(np.rad2deg(deltaw)) > TOL and max(abs(deltav) > TOL) and iterations<=max_loop_iter :

        iterations += 1

        # — build C, D—
        C = matC_caliter(psi, phi, Q_prev, w_prev, rho, v_prev, d1, P, c)
        D = matD_caliter(C, alpha, theta, psi, phi,
                Q_prev, w_prev, rho, v_prev, d1, P, c)
        #Qw,_,_,_ = np.linalg.lstsq(C, D, rcond=None)
        Qw = np.linalg.pinv(C) @ D
        Qn = Qw[0]
        wn = Qw[1]
        
        # — build A, B and update Q, w, v via GAMP —
        A      = matA_caliter(alpha, theta, psi, phi, Q_prev, w_prev, P)
        B  = matB_caliter(psi, phi, Q_prev, w_prev, rho,   d1, P, c)
        #vn,_,_,_ = np.linalg.lstsq(A, B)
        vn = np.linalg.pinv(A) @ B
        
        # — HV estimate —
        xh = vn[0]*np.sin(alpha[0])*np.cos(theta[0]) \
        - (d1 - vn[0])*np.sin(psi[0]+Qn)*np.cos(phi[0]+wn)
        yh = vn[0]*np.sin(alpha[0])*np.sin(theta[0]) \
        - (d1 - vn[0])*np.sin(psi[0]+Qn)*np.sin(phi[0]+wn)
        zh = vn[0]*np.cos(alpha[0]) \
        - (d1 - vn[0])*np.cos(psi[0]+Qn)
        HVh = np.array([xh,yh,zh])

        # — logging —
        history['Q'].append(Qn)
        history['w'].append(wn)
        history['HV'].append(HVh)
        history['error'].append(np.linalg.norm(HV-HVh))

        if iterprint==1:
            print(f"Iter {iterations:2d} | Q={np.rad2deg(Qn):6.3f}°, ω={np.rad2deg(wn):6.3f}°"
                f" | HV=({xh:6.2f},{yh:6.2f},{zh:6.2f})")
            print("====================================================")

        deltaQ = Qn - Q_prev
        deltaw = wn - w_prev
        deltav = vn - v_prev
        Q_prev = Qn
        w_prev = wn
        v_prev = vn
        iter_time = time.time() - start
        history['accumulated_time'].append(iter_time)
        
    history['iter'].append(iterations)
    end = time.time()
    elasped_inv_1 = end - start
    iterations = len(history['Q'])
    
    return history, elasped_inv_1, iterations

def loop_step_for_inverse_2(Q0, w0, v0, TOL, alpha, theta, psi, phi, tdoa, var_tdoa, rho, d1, P, c, iterprint):
    Q_prev, w_prev, v_prev = Q0, w0, v0
    history = {'Q':[], 'w':[], 'HV':[], 'iter':[], 'accumulated_time' : [], 'error' : []}
    iterations = 0
    deltaQ, deltaw = 1.0, 1.0
    deltav = np.ones(P+1)

    start = time.time()
    while abs(np.rad2deg(deltaQ))>TOL and abs(np.rad2deg(deltaw)) > TOL and max(abs(deltav) > TOL) and iterations<=max_loop_iter :

        iterations += 1

        # — build C, D—
        C = matC_caliter(psi, phi, Q_prev, w_prev, rho, v_prev, d1, P, c)
        var_C = 1e-4 * np.ones(3*(P-1))
        D = matD_caliter(C, alpha, theta, psi, phi,
                Q_prev, w_prev, rho, v_prev, d1, P, c)
        var_D = 1e-4 * np.ones(2*(P-1))
        
        # — build A, B and update Q, w, v via GAMP —
        A      = matA_caliter(alpha, theta, psi, phi, Q_prev, w_prev, P)
        z = matB_caliter(psi, phi, Q_prev, w_prev, tdoa,  d1, P, c)
        B  = matB_caliter(psi, phi, Q_prev, w_prev, rho,   d1, P, c)
        var_B   = matB_calvar(psi, phi, Q_prev, w_prev, P, c, var_tdoa)
        M = np.zeros((8*(P-1),2+P))
        M[:3*(P-1), 2:]   = A    # top-right
        M[3*(P-1):,  :2]  = C    # bottom-left
        G = np.zeros((8*(P-1),))
        G[:3*(P-1)]   = B   # top-right
        G[3*(P-1):]  = D   # bottom-left
        #xn,_,_,_ = np.linalg.lstsq(M, G, rcond=None)
        xn = np.linalg.pinv(M) @ G
        Qn = xn[0]
        wn = xn[1]
        vn = xn[2:]
        
        # — HV estimate —
        xh = vn[0]*np.sin(alpha[0])*np.cos(theta[0]) \
        - (d1 - vn[0])*np.sin(psi[0]+Qn)*np.cos(phi[0]+wn)
        yh = vn[0]*np.sin(alpha[0])*np.sin(theta[0]) \
        - (d1 - vn[0])*np.sin(psi[0]+Qn)*np.sin(phi[0]+wn)
        zh = vn[0]*np.cos(alpha[0]) \
        - (d1 - vn[0])*np.cos(psi[0]+Qn)
        HVh = np.array([xh,yh,zh])

        # — logging —
        history['Q'].append(Qn)
        history['w'].append(wn)
        history['HV'].append(HVh)
        history['error'].append(np.linalg.norm(HV-HVh))

        if iterprint==1:
            print(f"Iter {iterations:2d} | Q={np.rad2deg(Qn):6.3f}°, ω={np.rad2deg(wn):6.3f}°"
                f" | HV=({xh:6.2f},{yh:6.2f},{zh:6.2f})")

        deltaQ = Qn - Q_prev
        deltaw = wn - w_prev
        deltav = vn - v_prev
        Q_prev = Qn
        w_prev = wn
        v_prev = vn
        iter_time = time.time() - start
        history['accumulated_time'].append(iter_time)
    
        
    history['iter'].append(iterations)
    end = time.time()
    elasped_inv_2 = end - start
    iterations = len(history['Q'])
    
    return history, elasped_inv_2, iterations