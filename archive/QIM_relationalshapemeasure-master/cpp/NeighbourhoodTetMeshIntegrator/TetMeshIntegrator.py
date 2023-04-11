import numpy as np

from .NeighbourhoodTetMeshIntegrator import NeighbourhoodTetMeshIntegrator

def TetMeshIntegrator(verts, elements, dists, x):
    verts = verts.astype(np.float64)
    elements = elements.astype(np.int64)
    dists = dists.astype(np.float64)
    x = x.astype(np.float64)

    y_volume = np.empty(x.shape, dtype=np.float64)
    y_area = np.empty(x.shape, dtype=np.float64)

    NeighbourhoodTetMeshIntegrator(verts, elements, dists, x, y_volume, y_area)

    return y_volume, y_area