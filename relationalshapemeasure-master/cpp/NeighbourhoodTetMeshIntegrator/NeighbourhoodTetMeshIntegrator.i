%module NeighbourhoodTetMeshIntegrator

%{
#define SWIG_FILE_WITH_INIT
#include "c_NeighbourhoodTetMeshIntegrator.h"
%}

%include "numpy.i"

%init %{
import_array();
%}

%apply (double* INPLACE_ARRAY2, int DIM1, int DIM2) {
       (double * verts, int num_verts, int d1)
};

%apply (long* INPLACE_ARRAY2, int DIM1, int DIM2) {
       (const long * const elements, int num_elements, int d2)
};

%apply (double* INPLACE_ARRAY1, int DIM1) {
       (double * dists, int num_dists),
       (const double * const x, int num_x),
       (double * y_volume, int num_y1),
       (double * y_area, int num_y2)
};

void NeighbourhoodTetMeshIntegrator(
    double * verts, int num_verts, int d1,
    const long * const elements, int num_elements, int d2,
    double * dists, int num_dists,
    const double * const x, int num_x,
    double * y_volume, int num_y1,
    double * y_area, int num_y2
    );