#include <kernel_files/macros_cl.cl>

__kernel void tea_leaf_init_diag
(__global       double * __restrict const Kx,
 __global       double * __restrict const Ky,
 double rx, double ry)
{
    __kernel_indexes;

    if (row >= (y_min + 1) - 1 && row <= (y_max + 1) + 1
    && column >= (x_min + 1) - 1 && column <= (x_max + 1) + 1)
    {
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

__kernel void tea_leaf_calc_residual
(__global const double * __restrict const u,
 __global const double * __restrict const u0,
 __global       double * __restrict const r,
 __global const double * __restrict const Kx,
 __global const double * __restrict const Ky)
{
    __kernel_indexes;

    if (/*row >= (y_min + 1) - 0 &&*/ row <= (y_max + 1) + 0
    && /*column >= (x_min + 1) - 0 &&*/ column <= (x_max + 1) + 0)
    {
        const double smvp = (1.0
            + (Ky[THARR2D(0, 1, 0)] + Ky[THARR2D(0, 0, 0)])
            + (Kx[THARR2D(1, 0, 0)] + Kx[THARR2D(0, 0, 0)]))*u[THARR2D(0, 0, 0)]
            - (Ky[THARR2D(0, 1, 0)]*u[THARR2D(0, 1, 0)] + Ky[THARR2D(0, 0, 0)]*u[THARR2D(0, -1, 0)])
            - (Kx[THARR2D(1, 0, 0)]*u[THARR2D(1, 0, 0)] + Kx[THARR2D(0, 0, 0)]*u[THARR2D(-1, 0, 0)]);

        r[THARR2D(0, 0, 0)] = u0[THARR2D(0, 0, 0)] - smvp;
    }
}
