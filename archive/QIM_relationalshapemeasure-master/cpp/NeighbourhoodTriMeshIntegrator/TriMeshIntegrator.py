import numpy as np

from .NeighbourhoodTriMeshIntegrator import NeighbourhoodTriMeshIntegrator

def TriMeshIntegrator(verts, faces, dists, x):
    verts = verts.astype(np.float64)
    faces = faces.astype(np.int64)
    dists = dists.astype(np.float64)
    x = x.astype(np.float64)

    y_area = np.empty(x.shape, dtype=np.float64)
    y_circumference = np.empty(x.shape, dtype=np.float64)

    NeighbourhoodTriMeshIntegrator(verts, faces, dists, x, y_area, y_circumference)

    return y_area, y_circumference