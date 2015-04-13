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
    __kernel_indexes;

    if (WITHIN_BOUNDS)
    {
        if (PRECONDITIONER == TL_PREC_JAC_BLOCK)
        {
            __local double r_l[BLOCK_SZ];
            __local double z_l[BLOCK_SZ];

            r_l[lid] = r[THARR2D(0, 0, 0)];

            barrier(CLK_LOCAL_MEM_FENCE);
            if (loc_row == 0)
            {
                block_solve_func(r_l, z_l, cp, bfp, Kx, Ky);
            }
            barrier(CLK_LOCAL_MEM_FENCE);

            sd[THARR2D(0, 0, 0)] = z_l[lid]/theta;
        }
        else if (PRECONDITIONER == TL_PREC_JAC_DIAG)
        {
            //z[THARR2D(0, 0, 0)] = r[THARR2D(0, 0, 0)]*Mi[THARR2D(0, 0, 0)];
            sd[THARR2D(0, 0, 0)] = r[THARR2D(0, 0, 0)]*Mi[THARR2D(0, 0, 0)]/theta;
        }
        else
        {
            sd[THARR2D(0, 0, 0)] = r[THARR2D(0, 0, 0)]/theta;
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
 __global       double * __restrict const sd,

 __global       double * __restrict const z,
 __global const double * __restrict const cp,
 __global const double * __restrict const bfp,
 __global const double * __restrict const Mi,
 __global const double * __restrict const Kx,
 __global const double * __restrict const Ky,

 __constant const double * __restrict const alpha,
 __constant const double * __restrict const beta,
 int step)
{
    __kernel_indexes;

    if (WITHIN_BOUNDS)
    {
        if (PRECONDITIONER == TL_PREC_JAC_BLOCK)
        {
            __local double r_l[BLOCK_SZ];
            __local double z_l[BLOCK_SZ];

            r_l[lid] = r[THARR2D(0, 0, 0)];

            barrier(CLK_LOCAL_MEM_FENCE);
            if (loc_row == 0)
            {
                block_solve_func(r_l, z_l, cp, bfp, Kx, Ky);
            }
            barrier(CLK_LOCAL_MEM_FENCE);

            sd[THARR2D(0, 0, 0)] = alpha[step]*sd[THARR2D(0, 0, 0)]
                                + beta[step]*z_l[lid];
        }
        else if (PRECONDITIONER == TL_PREC_JAC_DIAG)
        {
            //z[THARR2D(0, 0, 0)] = r[THARR2D(0, 0, 0)]*Mi[THARR2D(0, 0, 0)];
            sd[THARR2D(0, 0, 0)] = alpha[step]*sd[THARR2D(0, 0, 0)]
                                + beta[step]*r[THARR2D(0, 0, 0)]*Mi[THARR2D(0, 0, 0)];
        }
        else
        {
            sd[THARR2D(0, 0, 0)] = alpha[step]*sd[THARR2D(0, 0, 0)]
                                + beta[step]*r[THARR2D(0, 0, 0)];
        }
    }
}

