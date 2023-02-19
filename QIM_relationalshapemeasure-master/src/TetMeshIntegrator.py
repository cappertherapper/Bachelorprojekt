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
    from numba import njit
except ImportError:
    numba_available = False

def if_njit(function, **kwargs):
    if not numba_available:
        return function
    else:
        return njit()(function)

TOL = 1e-10

@if_njit
def copy_verts(verts, v_indices, ordering):

    t_verts = np.empty(12, dtype=verts.dtype)

    for j in range(4):
        vn = v_indices[ordering[j]]
        for i in range(3):
            t_verts[3*j + i] = verts[3*vn + i]

    return t_verts

@if_njit
def get_sorted_indices(dists):
    # make a python version at check implementation
    ordering = np.array([0,1,2,3], dtype=np.int32)

    for i in range(3):
        for j in range(3-i):
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

    A = np.empty((4,4))
    v = np.empty(4)

    for i in range(4):
        v[i] = dists[i]
        for j in range(4):
            if j < 3:
                A[i, j] = verts[i*3 + j]
            else:
                A[i, j] = 1.0

    if np.abs(np.linalg.det(A)) < TOL:
        # this will be caught later
        return 0., 0., 0.
    x = np.linalg.solve(A, v)

    a, b, c = x[:3]

    return a, b, c

@if_njit
def full_tri_area(verts):
    # cross product and multiply by last vector
    a = np.array([verts[3] - verts[0], verts[4]  - verts[1], verts[5]  - verts[2]])
    b = np.array([verts[6] - verts[0], verts[7]  - verts[1], verts[8]  - verts[2]])
    cx = a[1] * b[2] - b[1] * a[2]
    cy = b[0] * a[2] - a[0] * b[2]
    cz = a[0] * b[1] - b[0] * a[1]

    return np.sqrt(cx**2 + cy**2 + cz**2) / 2.0

@if_njit
def full_tet_volume(verts):
    # cross product and multiply by last vector
    a = np.array([verts[3] - verts[0], verts[4]  - verts[1], verts[5]  - verts[2]])
    b = np.array([verts[6] - verts[0], verts[7]  - verts[1], verts[8]  - verts[2]])
    c = np.array([verts[9] - verts[0], verts[10] - verts[1], verts[11] - verts[2]])
    cm_x = (b[1] * c[2] - c[1] * b[2]) * a[0]
    cm_y = (c[0] * b[2] - b[0] * c[2]) * a[1]
    cm_z = (b[0] * c[1] - c[0] * b[1]) * a[2]

    return np.abs(cm_x + cm_y + cm_z) / 6.0

def transform_tet(verts, dists):

    # Normalize so first vertice is at the origin
    for j in range(1,4):#(int j = 1; j < 4; ++j)
        for i in range(3): #(int i = 0; i < 3; ++i)
            verts[j*3 + i] -= verts[i]

    for i in range(3): # (int i = 0; i < 3; ++i)
        verts[i] = 0

    a, b, c = get_linear_params(verts, dists)
    #print('Python', a, b, c)

    gradient_norm = np.sqrt(a**2 + b**2 + c**2)
    scale = gradient_norm

    t = -np.arctan2(c, b)

    for i in range(1,4):
        # rotx (x is unchanged, rot y and z)
        verts[i*3+1], verts[i*3+2] = rotx(verts[i*3+1], verts[i*3+2], t)

    b, c = rotx(b, c, t)

    t = -np.arctan2(b, a)

    for i in range(1,4):
        # rotz (z is unchanged, rot x and y)
        verts[i*3], verts[i*3+1] = rotz(verts[i*3], verts[i*3+1], t)

    # Scale to h scale
    for i in range(4):
        verts[i*3] = verts[i*3] * gradient_norm + dists[0]

    return verts, scale

@if_njit
def h_midpoint(p0, p1, h):

    p01 = np.empty(3)

    K = np.array([p1[0] - p0[0], p1[1] - p0[1], p1[2] - p0[2]])

    h0 = p0[0]
    h1 = p1[0]
    a = h0
    b = 1.0/(h1 - h0)

    for i in range(3):# (int i = 0; i < 3; ++i)
        p01[i] = p0[i] + K[i]*(h - a)*b

    return p01

@if_njit
def volume_fast(v0, v1, v2, v3, h):

    p01 = h_midpoint(v0, v1, h)
    p02 = h_midpoint(v0, v2, h)
    p03 = h_midpoint(v0, v3, h)
    p12 = h_midpoint(v1, v2, h)
    p13 = h_midpoint(v1, v3, h)

    P_tet1 = np.array([v0[0],  v0[1],  v0[2],
                       p01[0], p01[1], p01[2],
                       p02[0], p02[1], p02[2],
                       p03[0], p03[1], p03[2]])

    P_tet2 = np.array([v1[0],  v1[1],  v1[2],
                       p01[0], p01[1], p01[2],
                       p12[0], p12[1], p12[2],
                       p13[0], p13[1], p13[2]])

    volume_1 = full_tet_volume(P_tet1)
    volume_2 = full_tet_volume(P_tet2)

    return volume_1 - volume_2

@if_njit
def volume_safe(v0, v1, v2, v3, h):

    p02 = h_midpoint(v0, v2, h)
    p03 = h_midpoint(v0, v3, h)
    p12 = h_midpoint(v1, v2, h)
    p13 = h_midpoint(v1, v3, h)

    P_tet1 = np.array([v0[0],  v0[1],  v0[2],
                       p02[0], p02[1], p02[2],
                       p03[0], p03[1], p03[2],
                       p12[0], p12[1], p12[2]])

    P_tet2 = np.array([v0[0],  v0[1],  v0[2],
                       p03[0], p03[1], p03[2],
                       p12[0], p12[1], p12[2],
                       p13[0], p13[1], p13[2]])

    P_tet3 = np.array([v0[0],  v0[1],  v0[2],
                       v1[0],  v1[1],  v1[2],
                       p12[0], p12[1], p12[2],
                       p13[0], p13[1], p13[2]])

    volume_1 = full_tet_volume(P_tet1)
    volume_2 = full_tet_volume(P_tet2)
    volume_3 = full_tet_volume(P_tet3)

    return volume_1 + volume_2 + volume_3

@if_njit
def sliced_tet_volume(verts, full_volume, i, h):

    v0 = np.array([verts[i*12 + 0], verts[i*12 +  1], verts[i*12 +  2]])
    v1 = np.array([verts[i*12 + 3], verts[i*12 +  4], verts[i*12 +  5]])
    v2 = np.array([verts[i*12 + 6], verts[i*12 +  7], verts[i*12 +  8]])
    v3 = np.array([verts[i*12 + 9], verts[i*12 + 10], verts[i*12 + 11]])

    if (h <= v0[0]):
        # VOL 0
        return 0.0
    elif (h <= v1[0]):
        # VOL 1

        p01 = h_midpoint(v0, v1, h)
        p02 = h_midpoint(v0, v2, h)
        p03 = h_midpoint(v0, v3, h)

        P = np.array([v0[0],  v0[1],  v0[2],
                      p01[0], p01[1], p01[2],
                      p02[0], p02[1], p02[2],
                      p03[0], p03[1], p03[2]])

        volume = full_tet_volume(P)

        return volume

    elif (h <= v2[0]):
        # VOL 2
        if (np.abs(v0[0] - v1[0]) < TOL):
            # Degenerate case
            volume = volume_safe(v0, v1, v2, v3, h)
            return volume
        else:
            # Non-degenerate case
            volume = volume_fast(v0, v1, v2, v3, h)

            return volume

    elif (h <= v3[0]):
        # VOL 3

        p03 = h_midpoint(v0, v3, h)
        p13 = h_midpoint(v1, v3, h)
        p23 = h_midpoint(v2, v3, h)

        P_remaining = np.array([v3[0],  v3[1],  v3[2],
                                p03[0], p03[1], p03[2],
                                p13[0], p13[1], p13[2],
                                p23[0], p23[1], p23[2]])

        volume_remaining = full_tet_volume(P_remaining)
        if np.isnan(volume_remaining):
            print('NAN in volume_remaining on " << {P_remaining} {h}')
        if np.isnan(full_volume):
            print('NAN in full_volume on {v0} {v1} {v2} {v3} {h}')
        return full_volume - volume_remaining
    else:
        # VOL 4
        return full_volume

@if_njit
def sliced_tet_area(verts, i, h):

    v0 = np.array([verts[i*12 + 0], verts[i*12 +  1], verts[i*12 +  2]])
    v1 = np.array([verts[i*12 + 3], verts[i*12 +  4], verts[i*12 +  5]])
    v2 = np.array([verts[i*12 + 6], verts[i*12 +  7], verts[i*12 +  8]])
    v3 = np.array([verts[i*12 + 9], verts[i*12 + 10], verts[i*12 + 11]])

    if h <= v0[0]:
        # AREA 0
        return 0.0
    elif h <= v1[0]:
        # AREA 1
        p01 = h_midpoint(v0, v1, h)
        p02 = h_midpoint(v0, v2, h)
        p03 = h_midpoint(v0, v3, h)

        P = np.array([p01[0], p01[1], p01[2],
                      p02[0], p02[1], p02[2],
                      p03[0], p03[1], p03[2]])

        return full_tri_area(P)

    elif (h <= v2[0]):
        # AREA 2
        p02 = h_midpoint(v0, v2, h)
        p03 = h_midpoint(v0, v3, h)
        p12 = h_midpoint(v1, v2, h)
        p13 = h_midpoint(v1, v3, h)

        P_1 = np.array([p02[0], p02[1], p02[2],
                        p03[0], p03[1], p03[2],
                        p12[0], p12[1], p12[2]])
        P_2 = np.array([p03[0], p03[1], p03[2],
                        p12[0], p12[1], p12[2],
                        p13[0], p13[1], p13[2]])

        return full_tri_area(P_1) + full_tri_area(P_2)

    elif h <= v3[0]:
        # AREA 3

        p03 = h_midpoint(v0, v3, h)
        p13 = h_midpoint(v1, v3, h)
        p23 = h_midpoint(v2, v3, h)

        P = np.array([p03[0], p03[1], p03[2],
                      p13[0], p13[1], p13[2],
                      p23[0], p23[1], p23[2]])

        return full_tri_area(P)
    else:
        # AREA 4
        return 0.0

def TetMeshIntegrator(verts, elements, dists, x, adaptive):

    num_verts = len(verts)
    num_elements = len(elements)
    num_dists = len(dists)

    t_verts = np.empty(num_elements * 12, dtype=float)
    t_scales = np.empty(num_elements, dtype=float)
    full_volumes = np.empty(num_elements, dtype=float)

    tet_h_bounds = np.empty(num_elements * 2, dtype=float)

    verts = verts.flatten()
    elements = elements.flatten()

    #pragma omp parallel for
    for i in range(num_elements):

        t_verts_tmp = np.empty(12, dtype=float)

        v_indices = np.empty(4, dtype=np.int32)
        t_dists_tmp = np.empty(4, dtype=float)

        for j in range(4):
            v_indices[j] = elements[i*4+j]
            t_dists_tmp[j] = dists[v_indices[j]]

        t_dists_tmp, ordering = get_sorted_indices(t_dists_tmp)
        t_verts_tmp = copy_verts(verts, v_indices, ordering)

        tet_h_bounds[i*2] = t_dists_tmp[0]
        tet_h_bounds[i*2+1] = t_dists_tmp[3]

        t_verts_tmp, t_scale = transform_tet(t_verts_tmp, t_dists_tmp)
        #print(t_verts_tmp)

        t_scales[i] = t_scale
        full_volumes[i] = full_tet_volume(t_verts_tmp)

        for j in range(12):
            t_verts[i*12 + j] = t_verts_tmp[j]

    if adaptive:
        x = np.sort(np.unique(np.concatenate([x, tet_h_bounds+TOL, tet_h_bounds-TOL])))

    num_x = len(x)

    y_volume = np.empty(num_x, dtype=float)
    y_area = np.empty(num_x, dtype=float)

    #pragma omp parallel for
    for j in range(num_x):

        y_volume[j] = 0.0
        y_area[j] = 0.0

        for i in range(num_elements):

            if t_scales[i] != 0:

                #print(t_scales[i])
                y_volume[j] += sliced_tet_volume(t_verts, full_volumes[i], i, x[j]) / t_scales[i]
                y_area[j] += sliced_tet_area(t_verts, i, x[j])

    return x, y_volume, y_area
