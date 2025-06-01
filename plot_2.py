import numpy as np
import matplotlib.pyplot as plt
import config

def plot_3d(sc, history):
    SV_sc1 = np.zeros((1000,3))
    SV_sc2 = np.zeros((1000,3))
    SV_sc3 = np.zeros((1000,3))
    SV_sc4 = np.zeros((1000,3))
    HV_sc1 = np.zeros((1000,3))
    HV_sc2 = np.zeros((1000,3))
    HV_sc3 = np.zeros((1000,3))
    HV_sc4 = np.zeros((1000,3))
    for i in range(1000):
        SV_sc1[i,:] = sc[0,:]*i/1000
        SV_sc2[i,:] = sc[1,:]*i/1000
        SV_sc3[i,:] = sc[2,:]*i/1000
        SV_sc4[i,:] = sc[3,:]*i/1000
        HV_sc1[i,:] = config.HV + (sc[0,:]-config.HV)*i/1000
        HV_sc2[i,:] = config.HV + (sc[1,:]-config.HV)*i/1000
        HV_sc3[i,:] = config.HV + (sc[2,:]-config.HV)*i/1000
        HV_sc4[i,:] = config.HV + (sc[3,:]-config.HV)*i/1000
        
    # Results        
    fig = plt.figure(figsize=(7,7))

    # — 3D scene —
    ax = fig.add_subplot(projection='3d')

    lim = np.max(np.abs(np.vstack([config.SV, config.HV, sc]))) * 1.2
    xs = np.linspace(-10, lim, 2)
    ys = np.linspace(-lim, lim, 2)
    Xs, Ys = np.meshgrid(xs, ys)
    Zs = np.zeros_like(Xs)
    ax.plot_surface(
        Xs, Ys, Zs,
        color='gray', alpha=0.3,
        edgecolor='none', shade=True
    )
    ax.scatter(*config.SV, s=100, c='r', label='SV')
    ax.scatter(*config.HV, s=100, c='k', label='true HV')
    ax.scatter(*history['HV'][-1], c='g', label='est HV')
    for s in sc:
        ax.scatter(*s, c='b', marker='x')
    est2 = np.array(history['HV'])
    ax.plot(est2[:,0], est2[:,1], est2[:,2], c='g', label='est path')
    ax.plot(SV_sc1[:,0], SV_sc1[:,1], SV_sc1[:,2], c='grey', linestyle='dashed')
    ax.plot(SV_sc2[:,0], SV_sc2[:,1], SV_sc2[:,2], c='grey', linestyle='dashed')
    ax.plot(SV_sc3[:,0], SV_sc3[:,1], SV_sc3[:,2], c='grey', linestyle='dashed')
    ax.plot(SV_sc4[:,0], SV_sc4[:,1], SV_sc4[:,2], c='grey', linestyle='dashed')
    ax.plot(HV_sc1[:,0], HV_sc1[:,1], HV_sc1[:,2], c='grey', linestyle='dashed')
    ax.plot(HV_sc2[:,0], HV_sc2[:,1], HV_sc2[:,2], c='grey', linestyle='dashed')
    ax.plot(HV_sc3[:,0], HV_sc3[:,1], HV_sc3[:,2], c='grey', linestyle='dashed')
    ax.plot(HV_sc4[:,0], HV_sc4[:,1], HV_sc4[:,2], c='grey', linestyle='dashed')
    ax.set_xlabel('X')
    ax.set_ylabel('Y')
    ax.set_zlabel('Z')
    ax.set_title('Trajectory of Estimation in 3D Plot')
    ax.legend(loc='upper left')