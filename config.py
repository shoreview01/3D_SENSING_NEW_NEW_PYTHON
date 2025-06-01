import numpy as np

# Global constants and parameters

pi = np.pi

# Physical constants
SPEED_OF_LIGHT = 3e8  # meters per second

# Simulation parameters
TOL = 1e-4            # convergence tolerance (degrees)
MAX_ITER = 100        # max iterations for loop_step

# Vehicle and scatterer geometry (example placeholders)
SV = [0.0, 0.0, 0.0]  # sensing vehicle coordinates
HV = [20.0, -15.0, 2.0]  # true hidden vehicle coordinates

'''[10.0, 0.0, 5 + 4*np.random.rand()],
    [17.5, 0.0, 5 + 4*np.random.rand()],
    [17.5, 10.0, 5 + 4*np.random.rand()]'''
    #[10.0, 10.0, 5 + 4*np.random.rand()]
    
# Scatterer coordinates: list of four 3D points
SCATTERERS = [
    [10.0, 0.0, 5 + 4*np.random.rand()],
    [17.5, 0.0, 5 + 4*np.random.rand()],
    [17.5, 10.0, 5 + 4*np.random.rand()],
    [10.0, 10.0, 5 + 4*np.random.rand()]
]

# True orientation states
Q_TRUE = 15.0
W_TRUE = 45.0

Q_TRUE = np.deg2rad(Q_TRUE)
W_TRUE = np.deg2rad(W_TRUE)
D_TRUE = []

# Number of paths/scatterers
P = len(SCATTERERS)

# First path distance (will be set after geometry) placeholder
D1 = None  # to be filled by angle_dist_setting