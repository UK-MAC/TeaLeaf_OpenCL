#include <kernel_files/macros_cl.cl>
#include <kernel_files/tea_block_jacobi.cl>

__kernel void tea_leaf_cheby_solve_init_p
(__global const double * __restrict const u,
 __global const double * __restrict const u0,
 __global       double * __restrict const p,
 __global       double * __restrict const r,
 __global       double * __restrict const w,
 __global const double * __restrict const cp,
 __global const double * __restrict const bfp,
 __global const double * __restrict const Mi,
 __global const double * __restrict const Kx,
 __global const double * __restrict const Ky,
 double theta, double rx, double ry)
{
    __kernel_indexes;

    if (WITHIN_BOUNDS)
    {
        w[THARR2D(0, 0, 0)] = (1.0
            + (Ky[THARR2D(0, 1, 0)] + Ky[THARR2D(0, 0, 0)])
            + (Kx[THARR2D(1, 0, 0)] + Kx[THARR2D(0, 0, 0)]))*u[THARR2D(0, 0, 0)]
            - (Ky[THARR2D(0, 1, 0)]*u[THARR2D(0, 1, 0)] + Ky[THARR2D(0, 0, 0)]*u[THARR2D(0, -1, 0)])
            - (Kx[THARR2D(1, 0, 0)]*u[THARR2D(1, 0, 0)] + Kx[THARR2D(0, 0, 0)]*u[THARR2D(-1, 0, 0)]);

        r[THARR2D(0, 0, 0)] = u0[THARR2D(0, 0, 0)] - w[THARR2D(0, 0, 0)];

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

            p[THARR2D(0, 0, 0)] = z_l[lid]/theta;
        }
        else if (PRECONDITIONER == TL_PREC_JAC_DIAG)
        {
            p[THARR2D(0, 0, 0)] = (Mi[THARR2D(0, 0, 0)]*r[THARR2D(0, 0, 0)])/theta;
        }
        else
        {
            p[THARR2D(0, 0, 0)] = r[THARR2D(0, 0, 0)]/theta;
        }
    }
}

__kernel void tea_leaf_cheby_solve_calc_u
(__global       double * __restrict const u,
 __global const double * __restrict const p)
{
    __kernel_indexes;

    if (WITHIN_BOUNDS)
    {
        u[THARR2D(0, 0, 0)] += p[THARR2D(0, 0, 0)];
    }
}

__kernel void tea_leaf_cheby_solve_calc_p
(__global const double * __restrict const u,
 __global const double * __restrict const u0,
 __global       double * __restrict const p,
 __global       double * __restrict const r,
 __global       double * __restrict const w,
 __global const double * __restrict const cp,
 __global const double * __restrict const bfp,
 __global const double * __restrict const Mi,
 __global const double * __restrict const Kx,
 __global const double * __restrict const Ky,
 __constant const double * __restrict const alpha,
 __constant const double * __restrict const beta,
 double rx, double ry, int step)
{
    __kernel_indexes;

    if (WITHIN_BOUNDS)
    {
        w[THARR2D(0, 0, 0)] = (1.0
            + (Ky[THARR2D(0, 1, 0)] + Ky[THARR2D(0, 0, 0)])
            + (Kx[THARR2D(1, 0, 0)] + Kx[THARR2D(0, 0, 0)]))*u[THARR2D(0, 0, 0)]
            - (Ky[THARR2D(0, 1, 0)]*u[THARR2D(0, 1, 0)] + Ky[THARR2D(0, 0, 0)]*u[THARR2D(0, -1, 0)])
            - (Kx[THARR2D(1, 0, 0)]*u[THARR2D(1, 0, 0)] + Kx[THARR2D(0, 0, 0)]*u[THARR2D(-1, 0, 0)]);

        r[THARR2D(0, 0, 0)] = u0[THARR2D(0, 0, 0)] - w[THARR2D(0, 0, 0)];

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

            p[THARR2D(0, 0, 0)] = alpha[step]*p[THARR2D(0, 0, 0)]
                            + beta[step]*z_l[lid];
        }
        else if (PRECONDITIONER == TL_PREC_JAC_DIAG)
        {
            p[THARR2D(0, 0, 0)] = alpha[step]*p[THARR2D(0, 0, 0)]
                                + beta[step]*Mi[THARR2D(0, 0, 0)]*r[THARR2D(0, 0, 0)];
        }
        else
        {
            p[THARR2D(0, 0, 0)] = alpha[step]*p[THARR2D(0, 0, 0)]
                            + beta[step]*r[THARR2D(0, 0, 0)];
        }
    }
}

