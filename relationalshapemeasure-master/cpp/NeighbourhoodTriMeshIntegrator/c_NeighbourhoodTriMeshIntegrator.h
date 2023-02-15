#ifndef C_NEIGHBOURHOODTRIMESHINTEGRATOR_HPP
#define C_NEIGHBOURHOODTRIMESHINTEGRATOR_HPP

//#define PI 3.141592653589793238462643383279

void NeighbourhoodTriMeshIntegrator(
    double * verts, int num_verts, int d1,
    long * faces, int num_faces, int d2,
    double * dists, int num_dists,
    double * x, int num_x,
    double * y_area, int num_y1,
    double * y_circumference, int num_y2
    );

#endif // C_NEIGHBOURHOODTRIMESHINTEGRATOR_HPP