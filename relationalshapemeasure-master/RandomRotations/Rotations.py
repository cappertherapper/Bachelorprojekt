import numpy as np

def axisAngleRot(x,t):
    ux = x[0]
    uy = x[1]
    uz = x[2]

    R = np.zeros((3,3))

    ct = np.cos(t)
    st = np.sin(t)
    omct = (1 - ct)

    R[0,:] = np.array([ct + ux**2*omct,    ux*uy*omct - uz*st, ux*uz*omct + uy*st])
    R[1,:] = np.array([uy*ux*omct + uz*st, ct + uy**2*omct, uy*uz*omct - ux*st])
    R[2,:] = np.array([uz*ux*omct - uy*st, uz*uy*omct + ux*st, ct + uz**2*omct])

    return R