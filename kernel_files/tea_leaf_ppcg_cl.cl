#include <kernel_files/macros_cl.cl>
#include <kernel_files/tea_block_jacobi.cl>

__kernel void tea_leaf_ppcg_solve_init_sd
(__global const double * __restrict const r,
 __global       double * __restrict const sd,

 __global       double * __restrict const z,
 __global       double * __restrict const cp,
 __global       double * __restrict const bfp,
 __global       double * __restrict const Mi,
 __global const double * __restrict const Kx,
 __global const double * __restrict const Ky,

 double theta)
{
    if (PRECONDITIONER == TL_PREC_JAC_BLOCK)
    {
        const size_t column = get_global_id(0);
        const size_t row = get_global_id(1)*JACOBI_BLOCK_SIZE + 2;

        if (row > y_max || column > x_max) return;

        __private double r_l[JACOBI_BLOCK_SIZE];

        for (int k = 0; k < BLOCK_TOP; k++)
        {
            r_l[k] = r[THARR2D(0, k, 0)];
        }

        block_solve_func(r_l, z, cp, bfp, Kx, Ky);

        for (int k = 0; k < BLOCK_TOP; k++)
        {
            sd[THARR2D(0, k, 0)] = z[THARR2D(0, k, 0)]/theta;
        }
    }
    else
    {
        __kernel_indexes;

        if (WITHIN_BOUNDS)
        {
            if (PRECONDITIONER == TL_PREC_JAC_DIAG)
            {
                z[THARR2D(0, 0, 0)] = r[THARR2D(0, 0, 0)]*Mi[THARR2D(0, 0, 0)];
                sd[THARR2D(0, 0, 0)] = z[THARR2D(0, 0, 0)]/theta;
            }
            else
            {
                sd[THARR2D(0, 0, 0)] = r[THARR2D(0, 0, 0)]/theta;
            }
        }
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
 __global       double * __restrict const z,
 __global const double * __restrict const Mi,
 __global       double * __restrict const sd,
 __constant const double * __restrict const alpha,
 __constant const double * __restrict const beta,
 int step)
{
    __kernel_indexes;

    if (WITHIN_BOUNDS)
    {
        // JAC_BLOCK will call block_solve before this function
        if (PRECONDITIONER == TL_PREC_JAC_BLOCK)
        {
            sd[THARR2D(0, 0, 0)] = alpha[step]*sd[THARR2D(0, 0, 0)]
                                + beta[step]*z[THARR2D(0, 0, 0)];
        }
        if (PRECONDITIONER == TL_PREC_JAC_DIAG)
        {
            z[THARR2D(0, 0, 0)] = r[THARR2D(0, 0, 0)]*Mi[THARR2D(0, 0, 0)];
            sd[THARR2D(0, 0, 0)] = alpha[step]*sd[THARR2D(0, 0, 0)]
                                + beta[step]*z[THARR2D(0, 0, 0)];
        }
        else
        {
            sd[THARR2D(0, 0, 0)] = alpha[step]*sd[THARR2D(0, 0, 0)]
                                + beta[step]*r[THARR2D(0, 0, 0)];
        }
    }
}

