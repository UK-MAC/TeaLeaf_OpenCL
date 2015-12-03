#include <kernel_files/macros_cl.cl>
#include <kernel_files/tea_block_jacobi.cl>

// TODO this might need to be switched round because of the fortran order of tile indexing

// special indexing to write into the correct position in the 'coarse' arrays
// NB: coarse arrays have no halo data! They are one contiguous block
#define DEFLATION_IDX \
    /* column */ \
    ((get_group_id(0)) \
    /* row */ \
    + get_group_id(1)*get_num_groups(0))

/*
 *  Deflation needs a reduction which only reduces within the work then then
 *  writes out to the 'coarse' grid with the reduced value from each work group.
 *  These should be launched with a work group size of 8x8 or something similar
 */
#if defined(CL_DEVICE_TYPE_GPU)

    // binary tree reduction
    #define DEFLATION_REDUCTION(in, out, operation)                           \
        barrier(CLK_LOCAL_MEM_FENCE);                               \
        for (int offset = BLOCK_SZ / 2; offset > 0; offset /= 2)    \
        {                                                           \
            if (lid < offset)                                       \
            {                                                       \
                in[lid] = operation(in[lid],                        \
                                    in[lid + offset]);              \
            }                                                       \
            barrier(CLK_LOCAL_MEM_FENCE);                           \
        }                                                           \
        if(!lid)                                                    \
        {                                                           \
            out[DEFLATION_IDX] = in[0]; \
        }

#elif defined(CL_DEVICE_TYPE_CPU)

    // loop in first thread
    #define DEFLATION_REDUCTION(in, out, operation)                       \
        barrier(CLK_LOCAL_MEM_FENCE);                           \
        if (0 == lid)                                           \
        {                                                       \
            for (int offset = 1; offset < BLOCK_SZ; offset++)   \
            {                                                   \
                in[0] = operation(in[0], in[offset]);           \
            }                                                   \
            out[DEFLATION_IDX] = in[0]; \
        }

#endif

__kernel void tea_leaf_dpcg_coarsen_matrix
(kernel_info_t kernel_info,
 __GLOBAL__ double * __restrict const Kx,
 __GLOBAL__ double * __restrict const Ky,
 __GLOBAL__ double * __restrict const Kx_coarse,
 __GLOBAL__ double * __restrict const Ky_coarse)
{
    __kernel_indexes;

    __SHARED__ double Kx_sum_shared[BLOCK_SZ];
    __SHARED__ double Ky_sum_shared[BLOCK_SZ];

    if (WITHIN_BOUNDS)
    {
        Kx_sum_shared[lid] = Kx[THARR2D(0, 0, 0)];
        Ky_sum_shared[lid] = Ky[THARR2D(0, 0, 0)];
    }
    else
    {
        Kx_sum_shared[lid] = 0.0;
        Ky_sum_shared[lid] = 0.0;
    }

    DEFLATION_REDUCTION(Kx_sum_shared, Kx_coarse, SUM);
    DEFLATION_REDUCTION(Ky_sum_shared, Ky_coarse, SUM);
}

__kernel void tea_leaf_dpcg_prolong_Z
(kernel_info_t kernel_info,
 __GLOBAL__ double * __restrict const z,
 __GLOBAL__ double * __restrict const t2_coarse)
{
    __kernel_indexes;

    if (WITHIN_BOUNDS)
    {
        z[THARR2D(0, 0, 0)] -= t2_coarse[DEFLATION_IDX];
    }
}

// these two are similar - could be reused

__kernel void tea_leaf_dpcg_subtract_u
(kernel_info_t kernel_info,
 __GLOBAL__ double * __restrict const u,
 __GLOBAL__ double * __restrict const t2_coarse)
{
    __kernel_indexes;

    if (WITHIN_BOUNDS)
    {
        u[THARR2D(0, 0, 0)] -= t2_coarse[DEFLATION_IDX];
    }
}

__kernel void tea_leaf_dpcg_restrict_ZT
(kernel_info_t kernel_info,
 __GLOBAL__ double * __restrict const r,
 __GLOBAL__ double * __restrict const ZTr_coarse)
{
    __kernel_indexes;

    __SHARED__ double ZTr_sum_shared[BLOCK_SZ];

    if (WITHIN_BOUNDS)
    {
        ZTr_sum_shared[lid] = r[THARR2D(0, 0, 0)];
    }
    else
    {
        ZTr_sum_shared[lid] = 0.0;
    }

    DEFLATION_REDUCTION(ZTr_sum_shared, ZTr_coarse, SUM);
}

__kernel void tea_leaf_dpcg_matmul_ZTA
(kernel_info_t kernel_info,
 __GLOBAL__ const double * __restrict const z,
 __GLOBAL__ const double * __restrict const Kx,
 __GLOBAL__ const double * __restrict const Ky,
 __GLOBAL__       double * __restrict const ztaz_coarse)
{
    __kernel_indexes;

    __SHARED__ double ztaz_shared[BLOCK_SZ];
    ztaz_shared[lid] = 0.0;

    if (WITHIN_BOUNDS)
    {
        ztaz_shared[lid] = (1.0
            + (Ky[THARR2D(0, 1, 0)] + Ky[THARR2D(0, 0, 0)])
            + (Kx[THARR2D(1, 0, 0)] + Kx[THARR2D(0, 0, 0)]))*z[THARR2D(0, 0, 0)]
            - (Ky[THARR2D(0, 1, 0)]*z[THARR2D(0, 1, 0)] + Ky[THARR2D(0, 0, 0)]*z[THARR2D(0, -1, 0)])
            - (Kx[THARR2D(1, 0, 0)]*z[THARR2D(1, 0, 0)] + Kx[THARR2D(0, 0, 0)]*z[THARR2D(-1, 0, 0)]);
    }

    DEFLATION_REDUCTION(ztaz_shared, ztaz_coarse, SUM);
}

__kernel void tea_leaf_dpcg_init_p
(kernel_info_t kernel_info,
 __GLOBAL__ double * __restrict const p,
 __GLOBAL__ double * __restrict const z)
{
    __kernel_indexes;

    if (WITHIN_BOUNDS)
    {
        p[THARR2D(0, 0, 0)] = z[THARR2D(0, 0, 0)];
    }
}

__kernel void tea_leaf_dpcg_store_r
(kernel_info_t kernel_info,
 __GLOBAL__ double * __restrict const r,
 __GLOBAL__ double * __restrict const r_m1)
{
    __kernel_indexes;

    if (WITHIN_BOUNDS)
    {
        r_m1[THARR2D(0, 0, 0)] = r[THARR2D(0, 0, 0)];
    }
}

__kernel void tea_leaf_dpcg_calc_rrn
(kernel_info_t kernel_info,
 __GLOBAL__ const double * __restrict const r,
 __GLOBAL__ const double * __restrict const r_m1,
 __GLOBAL__ const double * __restrict const z,
 __GLOBAL__       double * __restrict const rrn)
{
    __kernel_indexes;

    __SHARED__ double rrn_shared[BLOCK_SZ];

    if (WITHIN_BOUNDS)
    {
        rrn_shared[lid] = (r[THARR2D(0, 0, 0)] - r_m1[THARR2D(0, 0, 0)])*z[THARR2D(0, 0, 0)];
    }
    else
    {
        rrn_shared[lid] = 0.0;
    }

    REDUCTION(rrn_shared, rrn, SUM)
}

__kernel void tea_leaf_dpcg_calc_p
(kernel_info_t kernel_info,
 __GLOBAL__ double * __restrict const p,
 __GLOBAL__ double * __restrict const z,
 double beta)
{
    __kernel_indexes;

    if (WITHIN_BOUNDS)
    {
        p[THARR2D(0, 0, 0)] = z[THARR2D(0, 0, 0)] + beta*p[THARR2D(0, 0, 0)];
    }
}

//__kernel void tea_leaf_dpcg_calc_zrnorm
// Can just call 2norm kernel here with the correct arguments

__kernel void tea_leaf_dpcg_solve_z
(kernel_info_t kernel_info,
 __GLOBAL__ const double * __restrict const r,
 __GLOBAL__       double * __restrict const z,
 __GLOBAL__ const double * __restrict const cp,
 __GLOBAL__ const double * __restrict const bfp,
 __GLOBAL__ const double * __restrict const Mi,
 __GLOBAL__ const double * __restrict const Kx,
 __GLOBAL__ const double * __restrict const Ky)
{
    __kernel_indexes;

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
            if (WITHIN_BOUNDS)
            {
                block_solve_func(kernel_info,r_l, z_l, cp, bfp, Kx, Ky);
            }
        }
        barrier(CLK_LOCAL_MEM_FENCE);

        if (WITHIN_BOUNDS)
        {
            z[THARR2D(0, 0, 0)] = z_l[lid];
        }
    }
    else if (WITHIN_BOUNDS)
    {
        z[THARR2D(0, 0, 0)] = r[THARR2D(0, 0, 0)]*Mi[THARR2D(0, 0, 0)];
    }
}

