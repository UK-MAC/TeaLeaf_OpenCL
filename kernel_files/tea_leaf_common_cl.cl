#include <kernel_files/macros_cl.cl>

__kernel void tea_leaf_init_diag
(__global       double * __restrict const Kx,
 __global       double * __restrict const Ky,
 __global       double * __restrict const diag,
 double rx, double ry)
{
    __kernel_indexes;

    if (row >= (y_min + 1) - 1 && row <= (y_max + 1) + 1
    && column >= (x_min + 1) - 1 && column <= (x_max + 1) + 1)
    {
        #if 0
        diag[THARR2D(0, 0, 0)] = (1.0
            + ry*(Ky[THARR2D(0, 1, 0)] + Ky[THARR2D(0, 0, 0)])
            + rx*(Kx[THARR2D(1, 0, 0)] + Kx[THARR2D(0, 0, 0)]));
        #endif
        Kx[THARR2D(0, 0, 0)] *= rx;
        Ky[THARR2D(0, 0, 0)] *= ry;
    }
}

__kernel void tea_leaf_finalise
(__global const double * __restrict const density1,
 __global const double * __restrict const u1,
 __global       double * __restrict const energy1)
{
    __kernel_indexes;

    if (/*row >= (y_min + 1) - 0 &&*/ row <= (y_max + 1) + 0
    && /*column >= (x_min + 1) - 0 &&*/ column <= (x_max + 1) + 0)
    {
        energy1[THARR2D(0, 0, 0)] = u1[THARR2D(0, 0, 0)]/density1[THARR2D(0, 0, 0)];
    }
}

