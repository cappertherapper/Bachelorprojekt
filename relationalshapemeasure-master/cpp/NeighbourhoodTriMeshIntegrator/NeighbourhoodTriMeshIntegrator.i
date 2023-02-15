%module NeighbourhoodTriMeshIntegrator

%{
#define SWIG_FILE_WITH_INIT
#include "c_NeighbourhoodTriMeshIntegrator.h"
%}

%include "numpy.i"

%init %{
import_array();
%}

%apply (double* INPLACE_ARRAY2, int DIM1, int DIM2) {
       (double * verts, int num_verts, int d1)
};

%apply (long* INPLACE_ARRAY2, int DIM1, int DIM2) {
       (long * faces, int num_faces, int d2)
};

%apply (double* INPLACE_ARRAY2, int DIM1, int DIM2) {
       (double * tet_verts, int num_tet_verts, int d3)
};

%apply (long* INPLACE_ARRAY2, int DIM1, int DIM2) {
       (long * tet_elems, int num_tet_verts, int d4)
};

%apply (long* INPLACE_ARRAY2, int DIM1, int DIM2) {
       (long * tet_faces, int num_tet_verts, int d5)
};

%apply (double* INPLACE_ARRAY1, int DIM1) {
       (double * dists, int num_dists),
       (double * x, int num_x),
       (double * y_area, int num_y1),
       (double * y_circumference, int num_y2)
};

void NeighbourhoodTriMeshIntegrator(
    double * verts, int num_verts, int d1,
    long * faces, int num_faces, int d2,
    double * dists, int num_dists,
    double * x, int num_x,
    double * y_area, int num_y1,
    double * y_circumference, int num_y2
    );
