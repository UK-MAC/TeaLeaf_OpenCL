#include <kernel_files/macros_cl.cl>
#include <kernel_files/tea_block_jacobi.cl>

/*
 *  Kernels used for conjugate gradient method
 */

__kernel void tea_leaf_cg_solve_init_p
(kernel_info_t kernel_info,
 __GLOBAL__       double * __restrict const p,
 __GLOBAL__       double * __restrict const r,
 __GLOBAL__       double * __restrict const z,
 __GLOBAL__       double * __restrict const Mi,
 __GLOBAL__       double * __restrict const rro)
{
    __kernel_indexes;

    __SHARED__ double rro_shared[BLOCK_SZ];
    rro_shared[lid] = 0.0;

    if (WITHIN_BOUNDS)
    {
        if (PRECONDITIONER == TL_PREC_JAC_BLOCK)
        {
            // z initialised when block_solve kernel is called before this
            p[THARR2D(0, 0, 0)] = z[THARR2D(0, 0, 0)];
        }
        else if (PRECONDITIONER == TL_PREC_JAC_DIAG)
        {
            z[THARR2D(0, 0, 0)] = r[THARR2D(0, 0, 0)]*Mi[THARR2D(0, 0, 0)];
            p[THARR2D(0, 0, 0)] = z[THARR2D(0, 0, 0)];
        }
        else if (PRECONDITIONER == TL_PREC_NONE)
        {
            p[THARR2D(0, 0, 0)] = r[THARR2D(0, 0, 0)];
        }

        rro_shared[lid] = r[THARR2D(0, 0, 0)]*p[THARR2D(0, 0, 0)];
    }

    REDUCTION(rro_shared, rro, SUM)
}

/* reduce rro */

__kernel void tea_leaf_cg_solve_calc_w
(kernel_info_t kernel_info,
 __GLOBAL__       double * __restrict const pw,
 __GLOBAL__ const double * __restrict const p,
 __GLOBAL__       double * __restrict const w,
 __GLOBAL__ const double * __restrict const Kx,
 __GLOBAL__ const double * __restrict const Ky)
{
    __kernel_indexes;

    __SHARED__ double pw_shared[BLOCK_SZ];
    pw_shared[lid] = 0.0;

    if (WITHIN_BOUNDS)
    {
        w[THARR2D(0, 0, 0)] = (1.0
            + (Ky[THARR2D(0, 1, 0)] + Ky[THARR2D(0, 0, 0)])
            + (Kx[THARR2D(1, 0, 0)] + Kx[THARR2D(0, 0, 0)]))*p[THARR2D(0, 0, 0)]
            - (Ky[THARR2D(0, 1, 0)]*p[THARR2D(0, 1, 0)] + Ky[THARR2D(0, 0, 0)]*p[THARR2D(0, -1, 0)])
            - (Kx[THARR2D(1, 0, 0)]*p[THARR2D(1, 0, 0)] + Kx[THARR2D(0, 0, 0)]*p[THARR2D(-1, 0, 0)]);
        
        pw_shared[lid] = p[THARR2D(0, 0, 0)]*w[THARR2D(0, 0, 0)];
    }

    REDUCTION(pw_shared, pw, SUM);
}

/* reduce pw */
  
__kernel void tea_leaf_cg_solve_calc_ur
(kernel_info_t kernel_info,
 double alpha,
 __GLOBAL__       double * __restrict const u,
 __GLOBAL__ const double * __restrict const p,
 __GLOBAL__       double * __restrict const r,
 __GLOBAL__ const double * __restrict const w,

 __GLOBAL__       double * __restrict const z,
 __GLOBAL__ const double * __restrict const cp,
 __GLOBAL__ const double * __restrict const bfp,
 __GLOBAL__ const double * __restrict const Mi,
 __GLOBAL__ const double * __restrict const Kx,
 __GLOBAL__ const double * __restrict const Ky,

 __GLOBAL__       double * __restrict const rrn)
{
    __kernel_indexes;

    __SHARED__ double rrn_shared[BLOCK_SZ];
    rrn_shared[lid] = 0.0;

    if (WITHIN_BOUNDS)
    {
        u[THARR2D(0, 0, 0)] += alpha*p[THARR2D(0, 0, 0)];
        r[THARR2D(0, 0, 0)] -= alpha*w[THARR2D(0, 0, 0)];
    }

    if (PRECONDITIONER == TL_PREC_JAC_BLOCK)
    {
        __SHARED__ double z_l[BLOCK_SZ];

        if (WITHIN_BOUNDS)
        {
            rrn_shared[lid] = r[THARR2D(0, 0, 0)];
        }

        barrier(CLK_LOCAL_MEM_FENCE);
        if (loc_row == 0)
        {
            block_solve_func(kernel_info,rrn_shared, z_l, cp, bfp, Kx, Ky);
        }
        barrier(CLK_LOCAL_MEM_FENCE);

        if (WITHIN_BOUNDS)
        {
            z[THARR2D(0, 0, 0)] = z_l[lid];

            rrn_shared[lid] = rrn_shared[lid]*z_l[lid];
        }
    }
    else if (WITHIN_BOUNDS)
    {
        if (PRECONDITIONER == TL_PREC_JAC_DIAG)
        {
            z[THARR2D(0, 0, 0)] = r[THARR2D(0, 0, 0)]*Mi[THARR2D(0, 0, 0)];
            rrn_shared[lid] = r[THARR2D(0, 0, 0)]*z[THARR2D(0, 0, 0)];
        }
        else if (PRECONDITIONER == TL_PREC_NONE)
        {
            rrn_shared[lid] = r[THARR2D(0, 0, 0)]*r[THARR2D(0, 0, 0)];
        }
    }

    REDUCTION(rrn_shared, rrn, SUM);
}

/* reduce rrn */

__kernel void tea_leaf_cg_solve_calc_p
(kernel_info_t kernel_info,
 double beta,
 __GLOBAL__       double * __restrict const p,
 __GLOBAL__ const double * __restrict const r,
 __GLOBAL__ const double * __restrict const z)
{
    __kernel_indexes;

    if (WITHIN_BOUNDS)
    {
        if (PRECONDITIONER != TL_PREC_NONE)
        {
            p[THARR2D(0, 0, 0)] = z[THARR2D(0, 0, 0)] + beta*p[THARR2D(0, 0, 0)];
        }
        else
        {
            p[THARR2D(0, 0, 0)] = r[THARR2D(0, 0, 0)] + beta*p[THARR2D(0, 0, 0)];
        }
    }
}

