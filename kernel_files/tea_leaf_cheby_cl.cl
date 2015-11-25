#include <kernel_files/macros_cl.cl>
#include <kernel_files/tea_block_jacobi.cl>

__kernel void tea_leaf_cheby_solve_init_p
(kernel_info_t kernel_info,
 __GLOBAL__ const double * __restrict const u,
 __GLOBAL__ const double * __restrict const u0,
 __GLOBAL__       double * __restrict const p,
 __GLOBAL__       double * __restrict const r,
 __GLOBAL__       double * __restrict const w,
 __GLOBAL__ const double * __restrict const cp,
 __GLOBAL__ const double * __restrict const bfp,
 __GLOBAL__ const double * __restrict const Mi,
 __GLOBAL__ const double * __restrict const Kx,
 __GLOBAL__ const double * __restrict const Ky,
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
    }

    if (PRECONDITIONER == TL_PREC_JAC_BLOCK)
    {
        __SHARED__ double r_l[BLOCK_SZ];
        __SHARED__ double z_l[BLOCK_SZ];

        r_l[lid] = 0;
        z_l[lid] = 0;

        if (WITHIN_BOUNDS)
        {
            r_l[lid] = r[THARR2D(0, 0, 0)];
        }

        barrier(CLK_LOCAL_MEM_FENCE);
        if (loc_row == 0)
        {
            block_solve_func(kernel_info,r_l, z_l, cp, bfp, Kx, Ky);
        }
        barrier(CLK_LOCAL_MEM_FENCE);

        if (WITHIN_BOUNDS)
        {
            p[THARR2D(0, 0, 0)] = z_l[lid]/theta;
        }
    }
    else if (WITHIN_BOUNDS)
    {
        if (PRECONDITIONER == TL_PREC_JAC_DIAG)
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
(kernel_info_t kernel_info,
 __GLOBAL__       double * __restrict const u,
 __GLOBAL__ const double * __restrict const p)
{
    __kernel_indexes;

    if (WITHIN_BOUNDS)
    {
        u[THARR2D(0, 0, 0)] += p[THARR2D(0, 0, 0)];
    }
}

__kernel void tea_leaf_cheby_solve_calc_p
(kernel_info_t kernel_info,
 __GLOBAL__ const double * __restrict const u,
 __GLOBAL__ const double * __restrict const u0,
 __GLOBAL__       double * __restrict const p,
 __GLOBAL__       double * __restrict const r,
 __GLOBAL__       double * __restrict const w,
 __GLOBAL__ const double * __restrict const cp,
 __GLOBAL__ const double * __restrict const bfp,
 __GLOBAL__ const double * __restrict const Mi,
 __GLOBAL__ const double * __restrict const Kx,
 __GLOBAL__ const double * __restrict const Ky,
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
    }

    if (PRECONDITIONER == TL_PREC_JAC_BLOCK)
    {
        __SHARED__ double r_l[BLOCK_SZ];
        __SHARED__ double z_l[BLOCK_SZ];

        r_l[lid] = 0;
        z_l[lid] = 0;

        if (WITHIN_BOUNDS)
        {
            r_l[lid] = r[THARR2D(0, 0, 0)];
        }

        barrier(CLK_LOCAL_MEM_FENCE);
        if (loc_row == 0)
        {
            block_solve_func(kernel_info,r_l, z_l, cp, bfp, Kx, Ky);
        }
        barrier(CLK_LOCAL_MEM_FENCE);

        if (WITHIN_BOUNDS)
        {
            p[THARR2D(0, 0, 0)] = alpha[step]*p[THARR2D(0, 0, 0)]
                            + beta[step]*z_l[lid];
        }
    }
    else if (WITHIN_BOUNDS)
    {
        if (PRECONDITIONER == TL_PREC_JAC_DIAG)
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

