import numpy as np

def angle_dist_setting(SV, HV, SCATTERERS, c, Q_true, w_true):
    """
    Compute distances, angles of arrival/departure, TDoA, and noisy measurements.

    Args:
        SV: [x,y,z] of sensing vehicle
        HV: [x,y,z] of hidden vehicle
        SCATTERERS: list of P scatterer [x,y,z]
        c: speed of light
        Q_true, w_true: true orientation angles (radians)

    Returns:
        v: (P,) path lengths SV->scatterer
        d_true: (P,) total distances SV->scatterer->HV
        sc: scatterer list reordered by total distance
        alpha: (P,) AoA elevation angles
        theta: (P,) AoA azimuth angles
        psi: (P,) AoD elevation minus Q_true
        phi: (P,) AoD azimuth minus w_true
        tdoa: (P,) true time difference of arrival
        var_tdoa: variance for noisy TDoA
        rho: (P,) noisy TDoA measurements
    """
    P = len(SCATTERERS)
    sc = np.array(SCATTERERS)
    SV = np.array(SV)
    HV = np.array(HV)

    # distances
    v = np.linalg.norm(sc - SV, axis=1)
    d_v = np.linalg.norm(sc-HV, axis=1)
    d_true = v + d_v
    print(d_v)
    # reorder by increasing total distance
    order = np.argsort(d_true)
    sc = sc[order]
    v = v[order]
    d_true = d_true[order]

    # angles
    # AoA elevation alpha, AoA azimuth theta
    alpha = np.arctan2(np.linalg.norm(sc[:,:2], axis=1), sc[:,2])
    theta = np.arctan2(sc[:,1] - SV[1], sc[:,0] - SV[0])

    # AoD elevation psi, AoD azimuth phi
    rel = sc - HV
    psi = np.arctan2(np.linalg.norm(rel[:,:2], axis=1), rel[:,2]) - Q_true
    phi = np.arctan2(rel[:,1], rel[:,0]) - w_true

    # TDoA
    tdoa = (d_true - d_true[0]) / c
    var_tdoa = 2e-4 * np.mean(tdoa)**2  # placeholder or calibrated value
    rho = tdoa + np.sqrt(var_tdoa) * np.random.randn(P)

    return v, d_true, sc, alpha, theta, psi, phi, tdoa, var_tdoa, rho