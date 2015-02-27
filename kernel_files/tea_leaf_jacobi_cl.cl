#include <kernel_files/macros_cl.cl>

__kernel void tea_leaf_jacobi_copy_u
(__global const double * __restrict const u1,
 __global       double * __restrict const un)
{
    __kernel_indexes;

    if (/*row >= (y_min + 1) - 1 &&*/ row <= (y_max + 1) + 1
    && /*column >= (x_min + 1) - 1 &&*/ column <= (x_max + 1) + 1)
    {
        un[THARR2D(0, 0, 0)] = u1[THARR2D(0, 0, 0)];
    }
}

__kernel void tea_leaf_jacobi_solve
(__global const double * __restrict const Kx,
 __global const double * __restrict const Ky,
 __global const double * __restrict const u0,
 __global       double * __restrict const u1,
 __global const double * __restrict const un,
 __global       double * __restrict const error)
{
    __kernel_indexes;

    __local double error_local[BLOCK_SZ];
    error_local[lid] = 0;

    if (/*row >= (y_min + 1) &&*/ row <= (y_max + 1)
    && /*column >= (x_min + 1) &&*/ column <= (x_max + 1))
    {
        u1[THARR2D(0, 0, 0)] = (u0[THARR2D(0, 0, 0)]
            + Kx[THARR2D(1, 0, 0)]*un[THARR2D( 1,  0, 0)]
            + Kx[THARR2D(0, 0, 0)]*un[THARR2D(-1,  0, 0)]
            + Ky[THARR2D(0, 1, 0)]*un[THARR2D( 0,  1, 0)]
            + Ky[THARR2D(0, 0, 0)]*un[THARR2D( 0, -1, 0)])
            /(1.0 + (Kx[THARR2D(0, 0, 0)] + Kx[THARR2D(1, 0, 0)])
                  + (Ky[THARR2D(0, 0, 0)] + Ky[THARR2D(0, 1, 0)]));
        
        error_local[lid] = fabs(u1[THARR2D(0, 0, 0)] - un[THARR2D(0, 0, 0)]);
    }

    REDUCTION(error_local, error, SUM);
}

