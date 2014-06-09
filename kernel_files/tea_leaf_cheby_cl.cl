#include <kernel_files/macros_cl.cl>

__kernel void tea_leaf_cheby_solve_init_p
(__global const double * __restrict const u,
 __global const double * __restrict const u0,
 __global       double * __restrict const p,
 __global       double * __restrict const r,
 __global const double * __restrict const Mi,
 __global       double * __restrict const w,
 __global const double * __restrict const Kx,
 __global const double * __restrict const Ky,
 double theta, double rx, double ry)
{
    __kernel_indexes;

    if (/*row >= (y_min + 1) - 0 &&*/ row <= (y_max + 1) + 0
    && /*column >= (x_min + 1) - 0 &&*/ column <= (x_max + 1) + 0)
    {
        w[THARR2D(0, 0, 0)] = (1.0
            + (Ky[THARR2D(0, 1, 0)] + Ky[THARR2D(0, 0, 0)])
            + (Kx[THARR2D(1, 0, 0)] + Kx[THARR2D(0, 0, 0)]))*u[THARR2D(0, 0, 0)]
            - (Ky[THARR2D(0, 1, 0)]*u[THARR2D(0, 1, 0)] + Ky[THARR2D(0, 0, 0)]*u[THARR2D(0, -1, 0)])
            - (Kx[THARR2D(1, 0, 0)]*u[THARR2D(1, 0, 0)] + Kx[THARR2D(0, 0, 0)]*u[THARR2D(-1, 0, 0)]);

        r[THARR2D(0, 0, 0)] = u0[THARR2D(0, 0, 0)] - w[THARR2D(0, 0, 0)];
        p[THARR2D(0, 0, 0)] = (Mi[THARR2D(0, 0, 0)]*r[THARR2D(0, 0, 0)])/theta;
    }
}

__kernel void tea_leaf_cheby_solve_calc_u
(__global       double * __restrict const u,
 __global const double * __restrict const p)
{
    __kernel_indexes;

    if (/*row >= (y_min + 1) - 0 && */row <= (y_max + 1) + 0
    && /*column >= (x_min + 1) - 0 &&*/ column <= (x_max + 1) + 0)
    {
        u[THARR2D(0, 0, 0)] += p[THARR2D(0, 0, 0)];
    }
}

__kernel void tea_leaf_cheby_solve_calc_p
(__global const double * __restrict const u,
 __global const double * __restrict const u0,
 __global       double * __restrict const p,
 __global       double * __restrict const r,
 __global       double * __restrict const Mi,
 __global       double * __restrict const w,
 __global const double * __restrict const Kx,
 __global const double * __restrict const Ky,
 __global const double * __restrict const alpha,
 __global const double * __restrict const beta,
 double rx, double ry, int step)
{
    __kernel_indexes;

    if (/*row >= (y_min + 1) - 0 &&*/ row <= (y_max + 1) + 0
    && /*column >= (x_min + 1) - 0 &&*/ column <= (x_max + 1) + 0)
    {
        w[THARR2D(0, 0, 0)] = (1.0
            + (Ky[THARR2D(0, 1, 0)] + Ky[THARR2D(0, 0, 0)])
            + (Kx[THARR2D(1, 0, 0)] + Kx[THARR2D(0, 0, 0)]))*u[THARR2D(0, 0, 0)]
            - (Ky[THARR2D(0, 1, 0)]*u[THARR2D(0, 1, 0)] + Ky[THARR2D(0, 0, 0)]*u[THARR2D(0, -1, 0)])
            - (Kx[THARR2D(1, 0, 0)]*u[THARR2D(1, 0, 0)] + Kx[THARR2D(0, 0, 0)]*u[THARR2D(-1, 0, 0)]);

        r[THARR2D(0, 0, 0)] = u0[THARR2D(0, 0, 0)] - w[THARR2D(0, 0, 0)];
        p[THARR2D(0, 0, 0)] = alpha[step]*p[THARR2D(0, 0, 0)]
                            + beta[step]*Mi[THARR2D(0, 0, 0)]*r[THARR2D(0, 0, 0)];
    }
}

__kernel void tea_leaf_cheby_solve_calc_resid
(__global const double * __restrict const r,
 __global       double * __restrict const rro)
{
    __kernel_indexes;

    __local double rro_shared[BLOCK_SZ];
    rro_shared[lid] = 0.0;

    if (/*row >= (y_min + 1) - 0 &&*/ row <= (y_max + 1) + 0
    && /*column >= (x_min + 1) - 0 &&*/ column <= (x_max + 1) + 0)
    {
        rro_shared[lid] = r[THARR2D(0, 0, 0)]*r[THARR2D(0, 0, 0)];
    }

    REDUCTION(rro_shared, rro, SUM)
}

