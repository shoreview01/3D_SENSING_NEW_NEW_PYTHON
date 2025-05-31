import numpy as np

def triop(opt, val1, val2):
    """
    Perform combined trig operations:
    opt=1: sin(val1)*cos(val2)
    opt=2: sin(val1)*sin(val2)
    opt=3: cos(val1)*cos(val2)
    opt=4: cos(val1)*sin(val2)
    """
    if opt == 1:
        return np.sin(val1) * np.cos(val2)
    elif opt == 2:
        return np.sin(val1) * np.sin(val2)
    elif opt == 3:
        return np.cos(val1) * np.cos(val2)
    elif opt == 4:
        return np.cos(val1) * np.sin(val2)
    else:
        raise ValueError(f"Invalid triop opt: {opt}")