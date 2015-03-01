#include <kernel_files/macros_cl.cl>

/*
 *  Kernels used for conjugate gradient method
 */

__kernel void tea_leaf_cg_solve_init_p
(__global       double * __restrict const p,
 __global       double * __restrict const r,
 __global       double * __restrict const z,
 __global       double * __restrict const rro)
{
    __kernel_indexes;

    __local double rro_shared[BLOCK_SZ];
    rro_shared[lid] = 0.0;

    if (WITHIN_BOUNDS)
    {
#if defined(USE_PRECONDITIONER)
        p[THARR2D(0, 0, 0)] = z[THARR2D(0, 0, 0)];
#else
        p[THARR2D(0, 0, 0)] = r[THARR2D(0, 0, 0)];
#endif

        rro_shared[lid] = r[THARR2D(0, 0, 0)]*p[THARR2D(0, 0, 0)];
    }

    REDUCTION(rro_shared, rro, SUM)
}

/* reduce rro */

__kernel void tea_leaf_cg_solve_calc_w
(__global       double * __restrict const pw,
 __global const double * __restrict const p,
 __global       double * __restrict const w,
 __global const double * __restrict const Kx,
 __global const double * __restrict const Ky)
{
    __kernel_indexes;

    __local double pw_shared[BLOCK_SZ];
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
(double alpha,
 __global       double * __restrict const u,
 __global const double * __restrict const p,
 __global       double * __restrict const r,
 __global const double * __restrict const w,

 __global       double * __restrict const z,
 __global       double * __restrict const cp,
 __global       double * __restrict const bfp,
 __global const double * __restrict const Kx,
 __global const double * __restrict const Ky,

 __global       double * __restrict const rrn)
{
#if defined(USE_PRECONDITIONER)
    const size_t column = get_global_id(0);
    const size_t row = get_global_id(1)*BLOCK_SIZE + 2;

    if (WITHIN_BOUNDS)
    {

#define COEF_A (1*(-Ky[THARR2D(0,j+ 0, 0)]))
#define COEF_B (1*(1.0 + (Ky[THARR2D(0,j+ 1, 0)] + Ky[THARR2D(0,j+ 0, 0)]) + (Kx[THARR2D(1,j+ 0, 0)] + Kx[THARR2D(0,j+ 0, 0)])))
#define COEF_C (1*(-Ky[THARR2D(0,j+ 1, 0)]))

        int j;
        __private double z_l[BLOCK_SIZE];
        __private double dp_l[BLOCK_SIZE];

        for (j = 0; j < BLOCK_SIZE; j++)
        {
            u[THARR2D(0, j, 0)] += alpha*p[THARR2D(0, j, 0)];
            r[THARR2D(0, j, 0)] -= alpha*w[THARR2D(0, j, 0)];
        }

        dp_l[j] = r[THARR2D(0, j, 0)]/COEF_B;

        for (j = 1; j < BLOCK_SIZE; j++)
        {
            dp_l[j] = (r[THARR2D(0, j, 0)] - COEF_A*dp_l[j - 1])*bfp[THARR2D(0, j, 0)];
        }

        j = BLOCK_SIZE - 1;

        z_l[j] = dp_l[j];

        for (j = BLOCK_SIZE - 2; j >= 0; j--)
        {
            z_l[j] = dp_l[j] - cp[THARR2D(0, j, 0)]*z_l[j + 1];
        }

        for (j = 0; j < BLOCK_SIZE; j++)
        {
            z[THARR2D(0, j, 0)] = z_l[j];
        }
    }
#else
    __kernel_indexes;

    __local double rrn_shared[BLOCK_SZ];
    rrn_shared[lid] = 0.0;

    if (WITHIN_BOUNDS)
    {
        u[THARR2D(0, 0, 0)] += alpha*p[THARR2D(0, 0, 0)];
        r[THARR2D(0, 0, 0)] -= alpha*w[THARR2D(0, 0, 0)];
        rrn_shared[lid] = r[THARR2D(0, 0, 0)]*r[THARR2D(0, 0, 0)];
    }

    REDUCTION(rrn_shared, rrn, SUM);
#endif
}

/* reduce rrn */

__kernel void tea_leaf_cg_solve_calc_p
(double beta,
 __global       double * __restrict const p,
 __global const double * __restrict const r,
 __global const double * __restrict const z)
{
    __kernel_indexes;

    if (WITHIN_BOUNDS)
    {
#if defined(USE_PRECONDITIONER)
        p[THARR2D(0, 0, 0)] = z[THARR2D(0, 0, 0)] + beta*p[THARR2D(0, 0, 0)];
#else
        p[THARR2D(0, 0, 0)] = r[THARR2D(0, 0, 0)] + beta*p[THARR2D(0, 0, 0)];
#endif
    }
}

