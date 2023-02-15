###############################################################################
#                                                                             #
#   Written by:    Hans Jacob Teglbjaerg Stephensen                           #
#   Date:          Nov 10th 2020                                              #
#                                                                             #
#   Please reference on use: https://arxiv.org/abs/2008.03927                 #
#   MIT license                                                               #
#                                                                             #
###############################################################################

import numpy as np

from .TriMeshIntegrator import TriMeshIntegrator
from .TetMeshIntegrator import TetMeshIntegrator

def measure_exterior(V, F, D, r, adaptive=False):
    """
    Computes the interior volume and interface area of a mesh in a given
    distance map

    Parameters
    ----------
    V : (N, 3) array_like
        Input vertex array.
    F : (M, 3) array_like
        Input triangular face array.
    D : (N) array_like
        Distance for each vertex
    r : (K) array_like
        Distance steps for which to measure the input elements

    Returns
    __________
    mu10 : (K) array_like
        The measure of surface area for each of the given distances
    mu11 : (K) array_like
        The measure of circumference of interface area for each of the given
        distances

    Examples
    --------
    >>> import numpy as np
    >>> from ReHaMeasure import measure_interior
    >>> np.random.seed(42)
    >>> V = np.random.uniform(0, 10, (100,3))
    >>> F = np.random.randint(0, 100, (100,3))
    >>> D = np.random.uniform(0,10,100)
    >>> r = np.linspace(0,15,100)
    >>> mu10, mu11 = measure_interior(V, F, D, r)
    """

    V = V.astype(float)
    D = D.astype(float)

    return TriMeshIntegrator(V, F, D, r, adaptive)

def measure_interior(V, T, D, r, adaptive=False):
    """
    Computes the interior volume and interface area of a mesh in a given
    distance map

    Parameters
    ----------
    V : (N, 3) array_like
        Input vertex array.
    T : (M, 4) array_like
        Input tetrahedral element array.
    D : (N) array_like
        Distance for each vertex
    r : (K) array_like
        Distance steps for which to measure the input elements

    Returns
    __________
    mu00 : (K) array_like
        The measure of volume for each of the given distances
    mu01 : (K) array_like
        The measure of interior interface area for each of the given distances

    Examples
    --------
    >>> import numpy as np
    >>> from ReHaMeasure import measure_interior
    >>> np.random.seed(42)
    >>> V = np.random.uniform(0, 10, (100,3))
    >>> T = np.random.randint(0, 100, (100,4))
    >>> D = np.random.uniform(0,10,100)
    >>> r = np.linspace(0,15,100)
    >>> mu00, mu01 = measure_interior(V, T, D, r)

    """

    V = V.astype(float)
    D = D.astype(float)

    return TetMeshIntegrator(V, T, D, r, adaptive)

