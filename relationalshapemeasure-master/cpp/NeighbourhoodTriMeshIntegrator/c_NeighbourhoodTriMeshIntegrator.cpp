#include "c_NeighbourhoodTriMeshIntegrator.h"
#include <cmath>
#include <algorithm>
#include <iostream>
#include <vector>
#include <boost/numeric/ublas/matrix.hpp>
#include <boost/numeric/ublas/vector.hpp>
#include <boost/numeric/ublas/lu.hpp>
#include <boost/numeric/ublas/io.hpp>

const double eps = 0.000000001;

void copy_verts(const double * const verts, const long * const v_indices, double * t_verts, const int * const ordering)
{
    for (int j = 0; j < 3; ++j)
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
    for (int j = 0; j < 2-i; ++j)
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

void get_linear_params(const double * const verts, const double * const dists, double &a, double &b)
{
    using namespace boost::numeric::ublas;

    matrix<double> A(3,3);
    vector<double> v(3);

    for (unsigned i = 0; i < 3; ++i) {
        v(i) = dists[i];
        for (unsigned j = 0; j < 3; ++j) {
            if (j < 2) A(i, j) = verts[i*3 + j];
            else A(i, j) = 1.0;
        }
    }

    permutation_matrix<size_t> pm(3);

    lu_factorize(A, pm);
    lu_substitute(A, pm, v);

    a = v(0);
    b = v(1);
}

void get_edge_linear_params(const double * const verts, double * linear_params, int i)
{
    const double x0 = verts[i*6];
    const double y0 = verts[i*6 + 1];
    const double x1 = verts[i*6 + 2];
    const double y1 = verts[i*6 + 3];
    const double x2 = verts[i*6 + 4];
    const double y2 = verts[i*6 + 5];

    linear_params[i*6] = (y1 - y0) / (x1 - x0);
    linear_params[i*6 + 1] = y0 - linear_params[i*6]*x0;
    linear_params[i*6 + 2] = (y2 - y0) / (x2 - x0);
    linear_params[i*6 + 3] = y0 - linear_params[i*6+2]*x0;
    linear_params[i*6 + 4] = (y2 - y1) / (x2 - x1);
    linear_params[i*6 + 5] = y1 - linear_params[i*6+4]*x1;
}

void transform_triangle(double * verts, double * dists, double & scale, double & full_area) // 9, 3
{
    // Normalize so first vertice is at the origin

    for (int i = 0; i < 3; ++i)
    {
        verts[3+i] -= verts[i];
        verts[6+i] -= verts[i];
        verts[i] = 0;
    }

    // Rotate so it corresponds to a flat triangWle

    double t;
    t = -std::atan2(verts[4], verts[3]);
    rotz(verts[3],verts[4], t);
    rotz(verts[6],verts[7], t);

    t = std::atan2(verts[5], verts[3]);
    roty(verts[3],verts[5], t);
    roty(verts[6],verts[8], t);

    t = -std::atan2(verts[8], verts[7]);
    rotx(verts[4],verts[5], t);
    rotx(verts[7],verts[8], t);

    // calculate full area of triangle
    const double ax = verts[0] - verts[3];
    const double ay = verts[1] - verts[4];
    const double bx = verts[0] - verts[6];
    const double by = verts[1] - verts[7];

    full_area = std::abs(ax*by - bx*ay) / 2.0;

    double a, b;
    if (full_area < eps) {
        verts[0] = dists[0]; verts[1] = dists[0]; verts[2] = dists[0]; // dont think we need this now we have the bounds elsewhere (Update: well, apparently we do...)
        return;
    }

    if (dists[0] == dists[2]) {
        verts[0] = dists[0]; verts[1] = dists[0]; verts[2] = dists[0]; // dont think we need this now we have the bounds elsewhere (Update: well, apparently we do...)
        return;
    }

    get_linear_params(verts, dists, a, b);

    double gradient_norm = std::sqrt(a*a + b*b);
    scale = 1.0/gradient_norm;

    // Change basis
    t = -std::atan2(b, a);
    rotz(verts[3],verts[4], t);
    rotz(verts[6],verts[7], t);

    // Scale to h scale
    verts[0] = verts[0] + dists[0];
    verts[3] = verts[3] * gradient_norm + dists[0];
    verts[6] = verts[6] * gradient_norm + dists[0];
}

double sliced_triangle_area(const double * const verts, const double * const edge_linear_params, const int i, double h)
{
    double x0 = verts[i*6 + 0];
    double y0 = verts[i*6 + 1];
    double x1 = verts[i*6 + 2];
    double y1 = verts[i*6 + 3];
    double x2 = verts[i*6 + 4];
    double y2 = verts[i*6 + 5];

    auto f01 = [=](double x) { return edge_linear_params[i*6 + 0]*x + edge_linear_params[i*6 + 1]; };
    auto f02 = [=](double x) { return edge_linear_params[i*6 + 2]*x + edge_linear_params[i*6 + 3]; };
    auto f12 = [=](double x) { return edge_linear_params[i*6 + 4]*x + edge_linear_params[i*6 + 5]; };

    if (x1 == x2 && y1 == y2) return 0.0;

    if (x0 == x1) {
        if (y0 == y1) return 0.0;
        if (x0 == x2) return 0.0;

        double height_full = x2 - x0;
        double base_full = y1 - y0;
        double full_area = std::abs(0.5*height_full*base_full);

        double right_height = x2 - h;
        double right_base = f02(h) - f12(h);
        double right_area = std::abs(0.5*right_height*right_base);

        double area = full_area - right_area;

        return area;
    }

    if (h < x1)
    {
        double height = h - x0;
        double base = f01(h) - f02(h);
        double area = std::abs(0.5*base*height);

        return std::abs(0.5*base*height);
    }

    double left_height = x1 - x0;
    double base = f01(x1) - f02(x1);
    double left_area = std::abs(base*left_height);

    double right_height = x2 - x1;
    double mid_base = y1 - f02(x1);
    double full_right_area = std::abs(right_height*mid_base);

    double remaining_right_area = std::abs((f12(h) - f02(h)) * (x2 - h));

    double area = 0.5*(left_area + full_right_area - remaining_right_area);

    return area;
}

double sliced_triangle_circ(const double * const verts, const double * const edge_linear_params, const int i, double h)
{
    double x0 = verts[i*6 + 0];
    double y0 = verts[i*6 + 1];
    double x1 = verts[i*6 + 2];
    double y1 = verts[i*6 + 3];
    double x2 = verts[i*6 + 4];
    double y2 = verts[i*6 + 5];

    if (x0 == x1 && y0 == y1) return 0.0;
    if (x0 == x2 && y0 == y2) return 0.0;
    if (x1 == x2 && y1 == y2) return 0.0;
    if (x0 == x1 && x1 == x2) return 0.0;
    if (y0 == y1 && y1 == y2) return 0.0;

    auto f02 = [=](double x) { return edge_linear_params[i*6 + 2]*x + edge_linear_params[i*6 + 3]; };
    if (h < x1)
    {
        auto f01 = [=](double x) { return edge_linear_params[i*6 + 0]*x + edge_linear_params[i*6 + 1]; };
        return std::abs(f01(h) - f02(h));
    }

    auto f12 = [=](double x) { return edge_linear_params[i*6 + 4]*x + edge_linear_params[i*6 + 5]; };
    return std::abs(f12(h) - f02(h));
}

void NeighbourhoodTriMeshIntegrator(
    double * verts, int num_verts, int d1,
    long * faces, int num_faces, int d2,
    double * dists, int num_dists,
    double * x, int num_x,
    double * y_area, int num_y1,
    double * y_circumference, int num_y2
    )
{
    double * t_verts = (double *) malloc(num_faces * 6 * sizeof(double)); // Every triangle, x0 y0 x1 y1 x2 y2
    double * t_scales = (double *) malloc(num_faces * sizeof(double));
    double * full_areas = (double *) malloc(num_faces * sizeof(double));
    double * triangle_h_bounds = (double *) malloc(num_faces * 2 * sizeof(double));
    double * edge_linear_params = (double *) malloc (num_faces * 6 * sizeof(double));

    #pragma omp parallel for
    for (int i = 0; i < num_faces; ++i)
    {

        double * t_verts_tmp = (double *) malloc(9*sizeof(double));

        int ordering[3] = {0, 1, 2};
        long v_indices[3] = {faces[i*3 + 0], faces[i*3 + 1], faces[i*3 + 2]};

        double t_dists_tmp[3] = {dists[v_indices[0]], dists[v_indices[1]], dists[v_indices[2]]};
        get_sorted_indices(t_dists_tmp, ordering);

        copy_verts(verts, v_indices, t_verts_tmp, ordering);

        triangle_h_bounds[i*2] = t_dists_tmp[0];
        triangle_h_bounds[i*2+1] = t_dists_tmp[2];

        double t_scale, full_area;

        transform_triangle(t_verts_tmp, t_dists_tmp, t_scale, full_area);

        for (int j = 0; j < 6; ++j)
        {
            int g_idx = (j / 2) * 3 + j%2; // This skips the z-coordinate which is now 0
            t_verts[i*6 + j] = t_verts_tmp[g_idx];
        }

        t_scales[i] = t_scale;
        full_areas[i] = full_area;

        get_edge_linear_params(t_verts, edge_linear_params, i);

        free(t_verts_tmp);
    }

    #pragma omp parallel for
    for (int j = 0; j < num_x; ++j)
    {
        y_area[j] = 0.0;
        y_circumference[j] = 0.0;

        for (int i = 0; i < num_faces; ++i)
        {
            if (x[j] <= triangle_h_bounds[i*2]) {
            } else if (x[j] >= triangle_h_bounds[i*2+1]) {
                y_area[j] += full_areas[i];
            } else {
                y_area[j] += sliced_triangle_area(t_verts, edge_linear_params, i, x[j]) * t_scales[i];
                y_circumference[j] += sliced_triangle_circ(t_verts, edge_linear_params, i, x[j]);
            }
        }
    }

    free(t_verts);
    free(t_scales);
    free(full_areas);
    free(triangle_h_bounds);
    free(edge_linear_params);
}
