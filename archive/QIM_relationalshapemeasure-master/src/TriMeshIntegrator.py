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

numba_available = True

try:
    from numba import jit, njit
except ImportError:
    numba_available = False

def if_njit(function, **kwargs):
    if not numba_available:
        return function
    else:
        return njit()(function)

def if_jit(function, **kwargs):
    if not numba_available:
        return function
    else:
        return jit()(function)

TOL = 1e-10

@if_njit
def copy_verts(verts, v_indices, ordering):

    t_verts = np.empty(9, dtype=verts.dtype)

    for j in range(3):
        vn = v_indices[ordering[j]]
        for i in range(3):
            t_verts[3*j + i] = verts[3*vn + i]

    return t_verts

@if_njit
def get_sorted_indices(dists):

    ordering = np.array([0,1,2], dtype=np.int32)

    for i in range(3):
        for j in range(2-i):
            if (dists[j] > dists[j+1]):
                dists[j], dists[j+1] = dists[j+1], dists[j]
                ordering[j], ordering[j+1] = ordering[j+1], ordering[j]

    return dists, ordering

@if_njit
def rotx(vy, vz, t):
    ct = np.cos(t)
    st = np.sin(t)

    tmp_vz = vy * st + vz * ct

    vy =  vy * ct - vz * st
    vz = tmp_vz

    return vy, vz

@if_njit
def roty(vx, vz, t):
    ct = np.cos(t)
    st = np.sin(t)

    tmp_vz = ct * vz - st * vx

    vx = vx * ct + st * vz
    vz = tmp_vz

    return vx, vz

@if_njit
def rotz(vx, vy, t):
    ct = np.cos(t)
    st = np.sin(t)

    tmp_vy = vx * st + vy * ct

    vx = vx * ct - vy * st
    vy = tmp_vy

    return vx, vy

def get_linear_params(verts, dists):

    A = np.empty((3,3))
    v = np.empty(3)

    for i in range(3):
        v[i] = dists[i]
        for j in range(3):
            if j < 2:
                A[i, j] = verts[i*3 + j]
            else:
                A[i, j] = 1.0

    if np.abs(np.linalg.det(A)) < TOL:
        # this will be caught later
        return 0., 0.

    x = np.linalg.solve(A, v)

    a, b = x[:2]

    return a, b

@if_njit
def get_edge_linear_params(verts, i):
    #x0, y0, x1, y1, x2, y2 = verts[i*6:(i+1)*6] # should work as well

    linear_params = np.zeros(6, dtype=np.float64)

    x0 = verts[i*6]
    y0 = verts[i*6 + 1]
    x1 = verts[i*6 + 2]
    y1 = verts[i*6 + 3]
    x2 = verts[i*6 + 4]
    y2 = verts[i*6 + 5]

    if np.abs(x1 - x0) < TOL:
        linear_params[0] = np.inf
    else:
        linear_params[0] = (y1 - y0) / (x1 - x0)
    linear_params[1] = y0 - linear_params[0]*x0
    if np.abs(x2 - x0) < TOL:
        linear_params[2] = np.inf
    else:
        linear_params[2] = (y2 - y0) / (x2 - x0)
    linear_params[3] = y0 - linear_params[2]*x0
    if np.abs(x2 - x1) < TOL:
        linear_params[4] = np.inf
    else:
        linear_params[4] = (y2 - y1) / (x2 - x1)
    linear_params[5] = y1 - linear_params[4]*x1

    return linear_params

def transform_triangle(verts, dists):

    # Normalize such that the first vertex is at the origin
    for i in range(3):

        verts[3+i] -= verts[i]
        verts[6+i] -= verts[i]
        verts[i] = 0

    # Rotate so it corresponds to a flat triangWle

    t = -np.arctan2(verts[4], verts[3])
    verts[3], verts[4] = rotz(verts[3],verts[4], t)
    verts[6], verts[7] = rotz(verts[6],verts[7], t)

    t = np.arctan2(verts[5], verts[3])
    verts[3], verts[5] = roty(verts[3],verts[5], t)
    verts[6], verts[8] = roty(verts[6],verts[8], t)

    t = -np.arctan2(verts[8], verts[7])
    verts[4], verts[5] = rotx(verts[4],verts[5], t)
    verts[7], verts[8] = rotx(verts[7],verts[8], t)

    # calculate full area of triangle
    ax = verts[0] - verts[3]
    ay = verts[1] - verts[4]
    bx = verts[0] - verts[6]
    by = verts[1] - verts[7]

    full_area = np.abs(ax*by - bx*ay) / 2.0

    if (full_area < TOL):
        verts[0] = dists[0]
        verts[1] = dists[0]
        verts[2] = dists[0] # dont think we need this now we have the bounds elsewhere (Update: well, apparently we do...)
        return verts, 1, full_area


    if (dists[0] == dists[2]):
        verts[0] = dists[0]
        verts[1] = dists[0]
        verts[2] = dists[0] # dont think we need this now we have the bounds elsewhere (Update: well, apparently we do...)
        return verts, 1, full_area

    a, b = get_linear_params(verts, dists)

    gradient_norm = np.sqrt(a*a + b*b)
    scale = 1.0/gradient_norm

    # Change basis
    t = -np.arctan2(b, a)
    verts[3],verts[4] = rotz(verts[3],verts[4], t)
    verts[6],verts[7] = rotz(verts[6],verts[7], t)

    # Scale to h scale
    verts[0] = verts[0] + dists[0]
    verts[3] = verts[3] * gradient_norm + dists[0]
    verts[6] = verts[6] * gradient_norm + dists[0]

    return verts, scale, full_area

def sliced_triangle_area(verts, edge_linear_params, i, h):

    #x0, y0, x1, y1, x2, y2 = verts[i*6:(i+1)*6] # should work as well

    x0 = verts[i*6 + 0]
    y0 = verts[i*6 + 1]
    x1 = verts[i*6 + 2]
    y1 = verts[i*6 + 3]
    x2 = verts[i*6 + 4]
    y2 = verts[i*6 + 5]

    if (x1 == x2) and (y1 == y2):
        return 0.0

    if x0 == x2:
        return 0.0

    if x0 == x1:
        if y0 == y1:
            return 0.0
        if x0 == x2:
            return 0.0

        f02 = lambda x: edge_linear_params[i*6 + 2]*x + edge_linear_params[i*6 + 3]
        f12 = lambda x: edge_linear_params[i*6 + 4]*x + edge_linear_params[i*6 + 5]

        height_full = x2 - x0
        base_full = y1 - y0
        full_area = np.abs(0.5*height_full*base_full)

        right_height = x2 - h
        right_base = f02(h) - f12(h)
        right_area = np.abs(0.5*right_height*right_base)

        area = full_area - right_area

        return area

    #print('linear params', edge_linear_params[i*6 + 0], edge_linear_params[i*6 + 1])
    if np.abs(edge_linear_params[i*6 + 0]) == np.inf or \
       np.abs(edge_linear_params[i*6 + 1]) == np.inf:
       f01 = lambda x: 0
    else:
        f01 = lambda x: edge_linear_params[i*6 + 0]*x + edge_linear_params[i*6 + 1]
    f02 = lambda x: edge_linear_params[i*6 + 2]*x + edge_linear_params[i*6 + 3]
    f12 = lambda x: edge_linear_params[i*6 + 4]*x + edge_linear_params[i*6 + 5]

    if h < x1:
        height = h - x0
        base = f01(h) - f02(h)
        area = np.abs(0.5*base*height)

        return np.abs(0.5*base*height)

    left_height = x1 - x0
    base = f01(x1) - f02(x1)

    left_area = np.abs(base*left_height)

    right_height = x2 - x1
    mid_base = y1 - f02(x1)

    full_right_area = np.abs(right_height*mid_base)

    remaining_right_area = np.abs((f12(h) - f02(h)) * (x2 - h))

    area = 0.5*(left_area + full_right_area - remaining_right_area)
    #print('area', area)
    return area

@if_njit
def sliced_triangle_circ(verts, edge_linear_params, i, h):

    #x0, y0, x1, y1, x2, y2 = verts[i*6:(i+1)*6] # should work as well

    x0 = verts[i*6 + 0]
    y0 = verts[i*6 + 1]
    x1 = verts[i*6 + 2]
    y1 = verts[i*6 + 3]
    x2 = verts[i*6 + 4]
    y2 = verts[i*6 + 5]

    if np.abs(x0 - x1) < TOL and np.abs(y0 - y1) < TOL:
        # degenerate flat triangle at left side
        return 0.0
    if np.abs(x0 - x2) < TOL and np.abs(y0 - y2) < TOL:
        # degenerate point triangle
        return 0.0
    if np.abs(x1 - x2) < TOL and np.abs(y1 - y2) < TOL:
        # degenerate flat triangle at right side
        return 0.0
    if np.abs(x0 - x1) < TOL and np.abs(x1 - x2) < TOL:
        # degenerate vertical triangle, the measure is not well defined here,
        # but will be 0.0 in the limit from both sides for this triangle
        return 0.0
    if np.abs(y0 - y1) < TOL and np.abs(y1 - y2) < TOL:
        # degenerate case where we have a horizontal flat triangle
        return 0.0

    f02 = lambda x: edge_linear_params[i*6 + 2]*x + edge_linear_params[i*6 + 3]

    if (h < x1):
        f01 = lambda x: edge_linear_params[i*6 + 0]*x + edge_linear_params[i*6 + 1]
        return np.abs(f01(h) - f02(h))

    f12 = lambda x: edge_linear_params[i*6 + 4]*x + edge_linear_params[i*6 + 5]

    return np.abs(f12(h) - f02(h))

def TriMeshIntegrator(verts, faces, dists, x, adaptive):


    num_verts = len(verts)
    num_faces = len(faces)
    num_dists = len(dists)

    t_verts = np.empty(num_faces * 6, dtype=float)
    t_scales = np.empty(num_faces, dtype=float)
    full_areas = np.empty(num_faces, dtype=float)
    triangle_h_bounds = np.empty(num_faces * 2, dtype=float)
    edge_linear_params = np.empty(num_faces * 6, dtype=float)

    verts = verts.flatten()
    faces = faces.flatten()

    #pragma omp parallel for
    for i in range(num_faces):

        v_indices = np.array([faces[i*3 + 0], faces[i*3 + 1], faces[i*3 + 2]])

        t_dists_tmp = np.array([dists[v_indices[0]], dists[v_indices[1]], dists[v_indices[2]]])

        t_dists_tmp, ordering = get_sorted_indices(t_dists_tmp)

        t_verts_tmp = copy_verts(verts, v_indices, ordering)

        triangle_h_bounds[i*2] = t_dists_tmp[0]
        triangle_h_bounds[i*2+1] = t_dists_tmp[2]

        t_verts_tmp, t_scale, full_area = transform_triangle(t_verts_tmp, t_dists_tmp)

        for j in range(6):
            g_idx = (j // 2) * 3 + j%2 # This skips the z-coordinate which is now 0
            t_verts[i*6 + j] = t_verts_tmp[g_idx]

        t_scales[i] = t_scale
        full_areas[i] = full_area

        edge_linear_params[i*6:(i+1)*6] = get_edge_linear_params(t_verts, i)

    if adaptive:
        x = np.sort(np.unique(np.concatenate([x, triangle_h_bounds+TOL, triangle_h_bounds-TOL])))

    num_x = len(x)

    y_area = np.empty(num_x, dtype=float)
    y_circumference = np.empty(num_x, dtype=float)

    #pragma omp parallel for
    for j in range(num_x):

        y_area[j] = 0.0
        y_circumference[j] = 0.0

        for i in range(num_faces):
            if x[j] <= triangle_h_bounds[i*2]:
                continue
            elif x[j] >= triangle_h_bounds[i*2+1]:
                y_area[j] += full_areas[i]
            else:
                y_area[j] += sliced_triangle_area(t_verts, edge_linear_params, i, x[j]) * t_scales[i]
                y_circumference[j] += sliced_triangle_circ(t_verts, edge_linear_params, i, x[j])

    return x, y_area, y_circumference
