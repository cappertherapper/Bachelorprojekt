#ifndef C_NEIGHBOURHOODTETMESHINTEGRATOR_HPP
#define C_NEIGHBOURHOODTETMESHINTEGRATOR_HPP

#define PI 3.141592653589793238462643383279

void NeighbourhoodTetMeshIntegrator(
    double * verts, int num_verts, int d1,
    const long * const elements, int num_elements, int d2,
    double * dists, int num_dists,
    const double * const x, int num_x,
    double * y_volume, int num_y1,
    double * y_area, int num_y2
    );

#endif // C_NEIGHBOURHOODTETMESHINTEGRATOR_HPP