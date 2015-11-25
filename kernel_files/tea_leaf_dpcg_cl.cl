#include <kernel_files/macros_cl.cl>
#include <kernel_files/tea_block_jacobi.cl>

// special indexing to write into the correct position in the 'coarse' arrays
#define DEFLATION_IDX \
    /* column */ \
    ((get_group_id(0) + get_global_offset(0)) \
    /* row */ \
    + (get_group_id(1) + get_global_offset(1)) \
    /* size of each row */ \
    *(get_num_groups(0) + 2*HALO_DEPTH))

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
    Kx_sum_shared[lid] = 0.0;
    Ky_sum_shared[lid] = 0.0;

    if (WITHIN_BOUNDS)
    {
        Kx_sum_shared[lid] = Kx[THARR2D(0, 0, 0)];
        Ky_sum_shared[lid] = Ky[THARR2D(0, 0, 0)];
    }

    DEFLATION_REDUCTION(Kx_sum_shared, Kx_coarse, SUM);
    DEFLATION_REDUCTION(Ky_sum_shared, Ky_coarse, SUM);
}

