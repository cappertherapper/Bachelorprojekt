#include "c_NeighbourhoodTetMeshIntegrator.h"
#include <cmath>
#include <algorithm>
#include <iostream>
#include <boost/numeric/ublas/matrix.hpp>
#include <boost/numeric/ublas/vector.hpp>
#include <boost/numeric/ublas/lu.hpp>
#include <boost/numeric/ublas/io.hpp>

#define TOL 1e-10

void copy_verts(const double * const verts, const long * const v_indices, double * t_verts, const int * const ordering)
{
    for (int j = 0; j < 4; ++j)
    {
        long vn = v_indices[ordering[j]];
        for (int i = 0; i < 3; ++i)
        {
            t_verts[3*j + i] = verts[3*vn + i];
        }
    }
}

void get_sorted_indices(double * dists, int * ordering)
{
    for (int i = 0; i < 3; ++i)
    for (int j = 0; j < 3-i; ++j) // not sure about this, verify with web for bubblesort
    {
        if (dists[j] > dists[j+1]) {
            std::swap(dists[j], dists[j+1]);
            std::swap(ordering[j], ordering[j+1]);
        }
    }
}

void rotx(double & vy, double & vz, const double t)
{
    const double ct = std::cos(t);
    const double st = std::sin(t);

    const double tmp_vz = vy * st + vz * ct;

    vy =  vy * ct - vz * st;
    vz = tmp_vz;
}

void roty(double & vx, double & vz, const double t)
{
    const double ct = std::cos(t);
    const double st = std::sin(t);

    const double tmp_vz = ct * vz - st * vx;

    vx = vx * ct + st * vz;
    vz = tmp_vz;
}

void rotz(double & vx, double & vy, const double t)
{
    const double ct = std::cos(t);
    const double st = std::sin(t);

    const double tmp_vy = vx * st + vy * ct;

    vx = vx * ct - vy * st;
    vy = tmp_vy;
}

void get_linear_params(const double * const verts, const double * const dists, double &a, double &b, double &c)
{
    using namespace boost::numeric::ublas;

    matrix<double> A(4,4);
    vector<double> v(4);

    for (unsigned i = 0; i < 4; ++i) {
        v(i) = dists[i];
        for (unsigned j = 0; j < 4; ++j) {
            if (j < 3) A(i, j) = verts[i*3 + j];
            else A(i, j) = 1.0;
        }
    }

    permutation_matrix<size_t> pm(4);

    int singular = lu_factorize(A, pm);

    if (singular) {
        a=0; b=0; c=0; // This is not my favorite solution, but it is caught later down the line
        return;
    }

    lu_substitute(A, pm, v);

    a = v(0);
    b = v(1);
    c = v(2);
}

double full_tri_area(const double * const verts)
{
    // cross product and multiply by last vector
    double a[3] = {verts[3] - verts[0], verts[4]  - verts[1], verts[5]  - verts[2]};
    double b[3] = {verts[6] - verts[0], verts[7]  - verts[1], verts[8]  - verts[2]};
    double cx = a[1] * b[2] - b[1] * a[2];
    double cy = b[0] * a[2] - a[0] * b[2];
    double cz = a[0] * b[1] - b[0] * a[1];

    return std::sqrt(cx*cx + cy*cy + cz*cz) / 2.0;
}

double full_tet_volume(const double * const verts)
{
    // cross product and multiply by last vector
    double a[3] = {verts[3] - verts[0], verts[4]  - verts[1], verts[5]  - verts[2]};
    double b[3] = {verts[6] - verts[0], verts[7]  - verts[1], verts[8]  - verts[2]};
    double c[3] = {verts[9] - verts[0], verts[10] - verts[1], verts[11] - verts[2]};
    double cm_x = (b[1] * c[2] - c[1] * b[2]) * a[0];
    double cm_y = (c[0] * b[2] - b[0] * c[2]) * a[1];
    double cm_z = (b[0] * c[1] - c[0] * b[1]) * a[2];

    return std::abs(cm_x + cm_y + cm_z) / 6.0;
}

void transform_tet(double * verts, double * dists, double & scale) // 9, 3
{
    // Normalize so first vertice is at the origin
    for (int j = 1; j < 4; ++j)
        for (int i = 0; i < 3; ++i)
            verts[j*3 + i] -= verts[i];

    for (int i = 0; i < 3; ++i)
        verts[i] = 0;

    double a, b, c;
    get_linear_params(verts, dists, a, b, c);

    double gradient_norm = std::sqrt(a*a + b*b + c*c);
    scale = gradient_norm;

    double t;

    t = -std::atan2(c, b);

    for (int i = 1; i < 4; ++i)
    {
        // rotx (x is unchanged, rot y and z)
        rotx(verts[i*3+1], verts[i*3+2], t);
    }
    rotx(b, c, t);

    t = -std::atan2(b, a);

    for (int i = 1; i < 4; ++i)
    {
        // rotz (z is unchanged, rot x and y)
        rotz(verts[i*3], verts[i*3+1], t);
    }

    // Scale to h scale
    for (int i = 0; i < 4; ++i)
    {
        verts[i*3] = verts[i*3] * gradient_norm + dists[0];
    }
}

void h_midpoint(const double * const p0, const double * const p1, const double h, double * p01)
{
    double K[3] = {p1[0] - p0[0], p1[1] - p0[1], p1[2] - p0[2]};

    double h0 = p0[0];
    double h1 = p1[0];
    double a = h0;
    double b = 1.0/(h1 - h0);

    for (int i = 0; i < 3; ++i)
    {
        p01[i] = p0[i] + K[i]*(h - a)*b;
    }
}

double volume_fast(const double * const v0, const double * const v1,
                   const double * const v2, const double * const v3,
                   const double h)
{
    double p01[3];
    h_midpoint(v0, v1, h, p01);
    double p02[3];
    h_midpoint(v0, v2, h, p02);
    double p03[3];
    h_midpoint(v0, v3, h, p03);
    double p12[3];
    h_midpoint(v1, v2, h, p12);
    double p13[3];
    h_midpoint(v1, v3, h, p13);

    double P_tet1[12] = { v0[0],  v0[1],  v0[2],
                         p01[0], p01[1], p01[2],
                         p02[0], p02[1], p02[2],
                         p03[0], p03[1], p03[2]};

    double P_tet2[12] = { v1[0],  v1[1],  v1[2],
                         p01[0], p01[1], p01[2],
                         p12[0], p12[1], p12[2],
                         p13[0], p13[1], p13[2]};

    double volume_1 = full_tet_volume(P_tet1);
    double volume_2 = full_tet_volume(P_tet2);

    return volume_1 - volume_2;
}

double volume_safe(const double * const v0, const double * const v1,
                   const double * const v2, const double * const v3,
                   const double h)
{
    double p02[3];
    h_midpoint(v0, v2, h, p02);
    double p03[3];
    h_midpoint(v0, v3, h, p03);
    double p12[3];
    h_midpoint(v1, v2, h, p12);
    double p13[3];
    h_midpoint(v1, v3, h, p13);

    double P_tet1[12] = { v0[0],  v0[1],  v0[2],
                         p02[0], p02[1], p02[2],
                         p03[0], p03[1], p03[2],
                         p12[0], p12[1], p12[2]};

    double P_tet2[12] = { v0[0],  v0[1],  v0[2],
                         p03[0], p03[1], p03[2],
                         p12[0], p12[1], p12[2],
                         p13[0], p13[1], p13[2]};

    double P_tet3[12] = { v0[0],  v0[1],  v0[2],
                          v1[0],  v1[1],  v1[2],
                         p12[0], p12[1], p12[2],
                         p13[0], p13[1], p13[2]};

    double volume_1 = full_tet_volume(P_tet1);
    double volume_2 = full_tet_volume(P_tet2);
    double volume_3 = full_tet_volume(P_tet3);

    return volume_1 + volume_2 + volume_3;
}

double sliced_tet_volume(const double * const verts, const double full_volume, const int i, double h)
{
    double v0[3] = {verts[i*12 + 0], verts[i*12 +  1], verts[i*12 +  2]};
    double v1[3] = {verts[i*12 + 3], verts[i*12 +  4], verts[i*12 +  5]};
    double v2[3] = {verts[i*12 + 6], verts[i*12 +  7], verts[i*12 +  8]};
    double v3[3] = {verts[i*12 + 9], verts[i*12 + 10], verts[i*12 + 11]};

    if (h <= v0[0]) {
        // VOL 0
        return 0.0;
    } else if (h <= v1[0]) {
        // VOL 1

        double p01[3];
        double p02[3];
        double p03[3];

        h_midpoint(v0, v1, h, p01);
        h_midpoint(v0, v2, h, p02);
        h_midpoint(v0, v3, h, p03);

        double P[12] = { v0[0],  v0[1],  v0[2],
                        p01[0], p01[1], p01[2],
                        p02[0], p02[1], p02[2],
                        p03[0], p03[1], p03[2]};

        double volume = full_tet_volume(P);

        if (std::isnan(volume))
        {
            std::cout << "NAN in volume (case 1) on " << P << " " << h << std::endl;
        }

        return volume;

    } else if (h <= v2[0]) {
        // VOL 2
        if (std::abs(v0[0] - v1[0]) < TOL) {
            // Degenerate case
            double volume = volume_safe(v0, v1, v2, v3, h);
            if (std::isnan(volume))
            {
                std::cout << "NAN in volume_safe on " << v0 << " " << v1 << " " << v2 << " " << v3 << " " << h << std::endl;
            }
            return volume;
        } else {
            // Non-degenerate case
            double volume = volume_fast(v0, v1, v2, v3, h);
            if (std::isnan(volume))
            {
                std::cout << "NAN in volume_fast on " << v0 << " " << v1 << " " << v2 << " " << v3 << " " << h << std::endl;
            }
            return volume;
        }
    } else if (h <= v3[0]) {
        // VOL 3
        double p03[3];
        double p13[3];
        double p23[3];

        h_midpoint(v0, v3, h, p03);
        h_midpoint(v1, v3, h, p13);
        h_midpoint(v2, v3, h, p23);

        double P_remaining[12] = { v3[0],  v3[1],  v3[2],
                                  p03[0], p03[1], p03[2],
                                  p13[0], p13[1], p13[2],
                                  p23[0], p23[1], p23[2]};

        double volume_remaining = full_tet_volume(P_remaining);
        if (std::isnan(volume_remaining))
        {
            std::cout << "NAN in volume_remaining on " << P_remaining << " " << h << std::endl;
        }
        if (std::isnan(full_volume))
        {
            std::cout << "NAN in full_volume on " << v0 << " " << v1 << " " << v2 << " " << v3 << " " << h << std::endl;
        }
        return full_volume - volume_remaining;
    } else {
        // VOL 4
        return full_volume;
    }
}

double sliced_tet_area(const double * const verts, const int i, double h)
{
    double v0[3] = {verts[i*12 + 0], verts[i*12 +  1], verts[i*12 +  2]};
    double v1[3] = {verts[i*12 + 3], verts[i*12 +  4], verts[i*12 +  5]};
    double v2[3] = {verts[i*12 + 6], verts[i*12 +  7], verts[i*12 +  8]};
    double v3[3] = {verts[i*12 + 9], verts[i*12 + 10], verts[i*12 + 11]};

    if (h <= v0[0]) {
        // AREA 0
        return 0.0;
    } else if (h <= v1[0]) {
        // AREA 1
        double p01[3];
        double p02[3];
        double p03[3];

        h_midpoint(v0, v1, h, p01);
        h_midpoint(v0, v2, h, p02);
        h_midpoint(v0, v3, h, p03);

        double P[9] = {p01[0], p01[1], p01[2],
                       p02[0], p02[1], p02[2],
                       p03[0], p03[1], p03[2]};

        return full_tri_area(P);

    } else if (h <= v2[0]) {
        // AREA 2
        double p02[3];
        double p03[3];
        double p12[3];
        double p13[3];

        h_midpoint(v0, v2, h, p02);
        h_midpoint(v0, v3, h, p03);
        h_midpoint(v1, v2, h, p12);
        h_midpoint(v1, v3, h, p13);

        double P_1[9] = {p02[0], p02[1], p02[2],
                         p03[0], p03[1], p03[2],
                         p12[0], p12[1], p12[2]};
        double P_2[9] = {p03[0], p03[1], p03[2],
                         p12[0], p12[1], p12[2],
                         p13[0], p13[1], p13[2]};

        return full_tri_area(P_1) + full_tri_area(P_2);

    } else if (h <= v3[0]) {
        // AREA 3
        double p03[3];
        double p13[3];
        double p23[3];

        h_midpoint(v0, v3, h, p03);
        h_midpoint(v1, v3, h, p13);
        h_midpoint(v2, v3, h, p23);

        double P[9] = {p03[0], p03[1], p03[2],
                       p13[0], p13[1], p13[2],
                       p23[0], p23[1], p23[2]};

        return full_tri_area(P);
    } else {
        // AREA 4
        return 0.0;
    }
}

void NeighbourhoodTetMeshIntegrator(
    double * verts, int num_verts, int d1,
    const long * const elements, int num_elements, int d2,
    double * dists, int num_dists,
    const double * const x, int num_x,
    double * y_volume, int num_y1,
    double * y_area, int num_y2
    )
{
    double * t_verts = (double *) malloc(num_elements * 12 * sizeof(double)); // Every triangle, x0 y0 x1 y1 x2 y2
    double * t_scales = (double *) malloc(num_elements * sizeof(double));
    double * full_volumes = (double *) malloc(num_elements * sizeof(double));

    #pragma omp parallel for
    for (int i = 0; i < num_elements; ++i)
    {

        double * t_verts_tmp = (double *) malloc(12*sizeof(double));

        int ordering[4];
        long v_indices[4];
        double t_dists_tmp[4];

        for (int j = 0; j < 4; ++j)
        {
            ordering[j] = j;
            v_indices[j] = elements[i*4+j];
            t_dists_tmp[j] = dists[v_indices[j]];
        }

        get_sorted_indices(t_dists_tmp, ordering);

        copy_verts(verts, v_indices, t_verts_tmp, ordering);

        double t_scale;

        transform_tet(t_verts_tmp, t_dists_tmp, t_scale);

        t_scales[i] = t_scale;
        full_volumes[i] = full_tet_volume(t_verts_tmp);

        for (int j = 0; j < 12; ++j)
        {
            t_verts[i*12 + j] = t_verts_tmp[j];
        }

        free(t_verts_tmp);
    }

    #pragma omp parallel for
    for (int j = 0; j < num_x; ++j)
    {
        y_volume[j] = 0.0;
        y_area[j] = 0.0;

        for (int i = 0; i < num_elements; ++i)
        {
            if (t_scales[i] != 0)
            {
                y_volume[j] += sliced_tet_volume(t_verts, full_volumes[i], i, x[j]) / t_scales[i];
                y_area[j] += sliced_tet_area(t_verts, i, x[j]);
            }
        }
    }

    free(t_verts);
    free(t_scales);
    free(full_volumes);
}