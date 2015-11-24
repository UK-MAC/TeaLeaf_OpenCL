#include "./kernel_files/macros_cl.cl"

#ifdef cl_khr_fp64
#pragma OPENCL EXTENSION cl_khr_fp64 : enable
#elif defined(cl_amd_fp64)
#pragma OPENCL EXTENSION cl_amd_fp64 : enable
#endif

/*
 *  Need to be defined:
 *  reduce_t = array type to reduce (double or int)
 *  REDUCE = operation to do
 *  INIT_RED_VAL = initial value before loading - eg, 0 for a sum
 *  LOCAL_SZ = local work group size, 1D
 */

#if defined(red_sum)
reduce_t REDUCE
(reduce_t a, reduce_t b)
{
    return SUM(a,b);
}
#elif defined (red_min)
reduce_t REDUCE
(reduce_t a, reduce_t b)
{
    return MIN(a,b);
}
#elif defined (red_max)
reduce_t REDUCE
(reduce_t a, reduce_t b)
{
    return MAX(a,b);
}
#else
    #error No definition for reduction
#endif

__kernel void reduction
(__global       reduce_t * const __restrict input)
{
    const int lid = get_local_id(0);
    const int gid = get_global_id(0);

    __local reduce_t scratch[LOCAL_SZ];

    // initialises to some initial value - different for different reductions
    scratch[lid] = INIT_RED_VAL;

    /*
     *  Read and write to two opposite halves of the reduction buffer so that
     *  there are no data races with reduction. eg, first stages reads from
     *  first half of buffer and writes results into second half, then second
     *  stage of reduction reads from second half and writes back into first,
     *  etc.
     */
    int dest_offset;
    int src_offset;

    if (RED_STAGE % 2)
    {
        src_offset = 0;
        dest_offset = ORIG_ELEMS_TO_REDUCE;
    }
    else
    {
        src_offset = ORIG_ELEMS_TO_REDUCE;
        dest_offset = 0;
    }

    for (int offset = 0; offset < SERIAL_REDUCTION_AMOUNT; offset++)
    {
        int read_idx =
            // size of serial reduction bit might be smaller than local size
            + (lid/SERIAL_REDUCTION_AMOUNT)*(SERIAL_REDUCTION_AMOUNT*SERIAL_REDUCTION_AMOUNT)
            // offset in this block
            + offset*SERIAL_REDUCTION_AMOUNT
            // each group gets local_sz*size of serial block to reduce
            + get_group_id(0)*(SERIAL_REDUCTION_AMOUNT*LOCAL_SZ)
            // and based on local id
            + lid%SERIAL_REDUCTION_AMOUNT
            ;

        if (read_idx < ELEMS_TO_REDUCE)
        {
            scratch[lid] = REDUCE(scratch[lid], input[read_idx + src_offset]);
        }
    }

    barrier(CLK_LOCAL_MEM_FENCE);

#if defined(CL_DEVICE_TYPE_GPU)

    for (int offset = LOCAL_SZ / 2; offset > 0; offset /= 2)
    {
        if (lid < offset)
        {
            scratch[lid] = REDUCE(scratch[lid], scratch[lid + offset]);
        }
        barrier(CLK_LOCAL_MEM_FENCE);
    }

#elif defined(CL_DEVICE_TYPE_CPU)

    //if (0 == lid)
    //{
    //    for (int offset = 1; offset < LOCAL_SZ; offset++)
    //    {
    //        scratch[0] = REDUCE(scratch[0], scratch[offset]);
    //    }
    //}
    scratch[lid] = work_group_reduce_add(scratch[lid]);

#elif defined(CL_DEVICE_TYPE_ACCELERATOR)

    // TODO special reductions for xeon phi in some fashion
    if (0 == lid)
    {
        for (int offset = 1; offset < LOCAL_SZ; offset++)
        {
            scratch[0] = REDUCE(scratch[0], scratch[offset]);
        }
    }

#else

    #error No device type specified for reduction

#endif

    if (0 == lid)
    {
#if (LOCAL_SZ == GLOBAL_SZ)
        // last stage - write back into 0 - no chance of data race
        input[0] = scratch[0];
#else
        input[dest_offset + get_group_id(0)] = scratch[0];
#endif
    }
}

