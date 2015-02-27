#include <kernel_files/macros_cl.cl>

__kernel void tea_leaf_ppcg_solve_init_sd
(__global const double * __restrict const r,
 __global const double * __restrict const Mi,
 __global       double * __restrict const sd,
 double theta)
{
    __kernel_indexes;

    if (WITHIN_BOUNDS)
    {
#if defined(USE_PRECONDITIONER)
        sd[THARR2D(0, 0, 0)] = (Mi[THARR2D(0, 0, 0)]*r[THARR2D(0, 0, 0)])/theta;
#else
        sd[THARR2D(0, 0, 0)] = r[THARR2D(0, 0, 0)]/theta;
#endif
    }
}

__kernel void tea_leaf_ppcg_solve_update_r
(__global       double * __restrict const u,
 __global       double * __restrict const r,
 __global const double * __restrict const Kx,
 __global const double * __restrict const Ky,
 __global       double * __restrict const sd)
{
    __kernel_indexes;

    if (WITHIN_BOUNDS)
    {
        const double result = (1.0
            + (Ky[THARR2D(0, 1, 0)] + Ky[THARR2D(0, 0, 0)])
            + (Kx[THARR2D(1, 0, 0)] + Kx[THARR2D(0, 0, 0)]))*sd[THARR2D(0, 0, 0)]
            - (Ky[THARR2D(0, 1, 0)]*sd[THARR2D(0, 1, 0)] + Ky[THARR2D(0, 0, 0)]*sd[THARR2D(0, -1, 0)])
            - (Kx[THARR2D(1, 0, 0)]*sd[THARR2D(1, 0, 0)] + Kx[THARR2D(0, 0, 0)]*sd[THARR2D(-1, 0, 0)]);

        r[THARR2D(0, 0, 0)] -= result;
        u[THARR2D(0, 0, 0)] += sd[THARR2D(0, 0, 0)];
    }
}

__kernel void tea_leaf_ppcg_solve_calc_sd
(__global const double * __restrict const r,
 __global const double * __restrict const Mi,
 __global       double * __restrict const sd,
 __constant const double * __restrict const alpha,
 __constant const double * __restrict const beta,
 int step)
{
    __kernel_indexes;

    if (WITHIN_BOUNDS)
    {
        sd[THARR2D(0, 0, 0)] = alpha[step]*sd[THARR2D(0, 0, 0)]
#if defined(USE_PRECONDITIONER)
#error Preconditioner does not yet work with ppcg solver - disable preconditioner in input file to run
                            + beta[step]*Mi[THARR2D(0, 0, 0)]*r[THARR2D(0, 0, 0)];
#else
                            + beta[step]*r[THARR2D(0, 0, 0)];
#endif
    }
}

