#include <kernel_files/macros_cl.cl>
#include <kernel_files/tea_block_jacobi.cl>

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
 __global const double * __restrict const cp,
 __global const double * __restrict const bfp,
 __global const double * __restrict const Kx,
 __global const double * __restrict const Ky,

 __global       double * __restrict const rrn)
{
    __local double rrn_shared[BLOCK_SZ];

#if defined(USE_PRECONDITIONER)
    const size_t column = get_global_id(0);
    const size_t row = get_global_id(1)*JACOBI_BLOCK_SIZE + 2;

    const size_t loc_column = get_local_id(0);
    const size_t loc_row = get_local_id(1);
    const size_t lid = loc_row*get_local_size(0) + loc_column;

    if (!lid)
    {
        for (int k = 0; k < BLOCK_TOP+1; k++)
        {
            rrn[(get_group_id(1)*get_num_groups(0) + get_group_id(0)) +
                get_num_groups(0)*get_num_groups(1)*k] = 0;
        }
    }

    rrn_shared[lid] = 0.0;

    if (row > y_max || column > x_max) return;

    __private double r_l[JACOBI_BLOCK_SIZE];

    for (int k = 0; k < BLOCK_TOP; k++)
    {
        u[THARR2D(0, k, 0)] += alpha*p[THARR2D(0, k, 0)];
        r_l[k] = r[THARR2D(0, k, 0)] -= alpha*w[THARR2D(0, k, 0)];
    }

    block_solve_func(r_l, z, cp, bfp, Kx, Ky);

    for (int k = 0; k < BLOCK_TOP; k++)
    {
        rrn_shared[lid] += z[THARR2D(0, k, 0)]*r_l[k];
    }
#else
    __kernel_indexes;

    rrn_shared[lid] = 0.0;

    if (WITHIN_BOUNDS)
    {
        u[THARR2D(0, 0, 0)] += alpha*p[THARR2D(0, 0, 0)];
        r[THARR2D(0, 0, 0)] -= alpha*w[THARR2D(0, 0, 0)];
        rrn_shared[lid] = r[THARR2D(0, 0, 0)]*r[THARR2D(0, 0, 0)];
    }
#endif

    REDUCTION(rrn_shared, rrn, SUM);
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

