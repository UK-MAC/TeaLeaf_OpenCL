#pragma OPENCL EXTENSION cl_khr_fp64 : enable

// preconditioners
#include "definitions.hpp"

#if defined(BLOCK_TOP_CHECK)
    #define BLOCK_TOP (MIN(((int)y_max + 2 - (int)row),(int)JACOBI_BLOCK_SIZE))
#else
    #define BLOCK_TOP (JACOBI_BLOCK_SIZE)
#endif

#define __kernel_indexes                            \
    const int column = get_global_id(0);			\
    const int row = get_global_id(1);				\
    const int loc_column = get_local_id(0);			\
    const int loc_row = get_local_id(1);			\
    const int lid = loc_row*LOCAL_X + loc_column;	\
    const int gid = row*get_global_size(0) + column;

#define THARR2D(x_offset, y_offset, big_row)                \
    ( (column + x_offset)                                   \
    + (row + y_offset)*(x_max + 2*HALO_DEPTH + big_row))

// check if within bounds, based on what was passed in when compiled - stops having to make sure 2 numbers in different places are the same
#define WITHIN_BOUNDS                               \
    (/*row >= (y_min + 1) - KERNEL_X_MIN &&*/       \
     row <= (y_max + HALO_DEPTH - 1) + KERNEL_Y_MAX &&           \
     /*column >= (x_min + 1) - KERNEL_Y_MIN &&*/    \
     column <= (x_max + HALO_DEPTH - 1) + KERNEL_X_MAX)
//if\s*(\/\?\*\?row >= ([^)]\+) \([-+] \w\)\?&&\*\?\/\? row <= ([^)]\+)\( [+-] \w\)\?\n\s\+&& \/\?\*\?column >= ([^)]\+) \([-+] \w \)\?&&\*\?\/\? column <= ([^)]\+)\( [+-] \w\)\?)

#ifdef CLOVER_NO_BUILTINS
    #define MAX(a,b) (a<b?a:b)
    #define MIN(a,b) (a>b?a:b)
    #define SUM(a,b) (a+b)
    #define SIGN(a,b) (((b) <  (0) && (a > (0))||((b) > (0) && ((a)<(0)))) ? (-a) : (a))
    #define SQRT(a) sqrt(convert_float(a))
#else
    #define MAX(a,b) max(a,b)
    #define MIN(a,b) min(a,b)
    #define SUM(a,b) ((a)+(b))
    #define SIGN(a,b) copysign(a,b)
    #define SQRT(a) sqrt(a)
#endif

// TODO probably can optimise reductions somehow
#if defined(CL_DEVICE_TYPE_GPU)

    // binary tree reduction
    #define REDUCTION(in, out, operation)                           \
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
            out[get_group_id(1)*get_num_groups(0) + get_group_id(0)] = in[0]; \
        }

#elif defined(CL_DEVICE_TYPE_CPU)

    // loop in first thread
    #define REDUCTION(in, out, operation)                       \
        barrier(CLK_LOCAL_MEM_FENCE);                           \
        if (0 == lid)                                           \
        {                                                       \
            for (int offset = 1; offset < BLOCK_SZ; offset++)   \
            {                                                   \
                in[0] = operation(in[0], in[offset]);           \
            }                                                   \
            out[get_group_id(1)*get_num_groups(0) + get_group_id(0)] = in[0]; \
        }

#elif defined(CL_DEVICE_TYPE_ACCELERATOR)

    // loop in first thread
    #define REDUCTION(in, out, operation)                       \
        barrier(CLK_LOCAL_MEM_FENCE);                           \
        if (0 == lid)                                           \
        {                                                       \
            for (int offset = 1; offset < BLOCK_SZ; offset++)   \
            {                                                   \
                in[0] = operation(in[0], in[offset]);           \
            }                                                   \
            out[get_group_id(1)*get_num_groups(0) + get_group_id(0)] = in[0]; \
        }

#if 0

    /*
     *  TODO
     *  
     *  8/16 wide vector units
     *  4 cores per thing
     *  57-61 cpus
     */

    #if 0
    #define REDUCTION(in, out, operation)                    \
        barrier(CLK_LOCAL_MEM_FENCE);                               \
        for (size_t offset = BLOCK_SZ / 2; offset > 0; offset /= 2) \
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
            out[get_group_id(1)*get_num_groups(0) + get_group_id(0)] = in[0]; \
        }
    #else
    #define REDUCTION(in, out, operation)                    \
    { \
        barrier(CLK_LOCAL_MEM_FENCE);                               \
        const size_t vecsz = 512/(sizeof(in[0])*8);  \
        const size_t redsz = vecsz*2; \
        size_t remain = BLOCK_SZ ; \
        do \
        { \
            if (!(lid % redsz) && lid < remain)                \
            {                                                           \
                /*for (size_t offset = 1; offset < redsz; offset++)    \
                {                                                       \
                    in[lid] = operation(in[lid], in[lid + offset]);               \
                }*/                                                       \
                /*in[0 + lid] = operation(in[0 + lid], in[0 + lid+vecsz]); \
                in[1 + lid] = operation(in[1 + lid], in[1 + lid+vecsz]); \
                in[2 + lid] = operation(in[2 + lid], in[2 + lid+vecsz]); \
                in[3 + lid] = operation(in[3 + lid], in[3 + lid+vecsz]); \
                in[4 + lid] = operation(in[4 + lid], in[4 + lid+vecsz]); \
                in[5 + lid] = operation(in[5 + lid], in[5 + lid+vecsz]); \
                in[6 + lid] = operation(in[6 + lid], in[6 + lid+vecsz]); \
                in[7 + lid] = operation(in[7 + lid], in[7 + lid+vecsz]);*/ \
                for (size_t offset = 0; offset < vecsz; offset++)    \
                {                                                       \
                    in[offset + lid] = operation(in[offset + lid], in[offset + lid+vecsz]); \
                }                                                       \
                barrier(CLK_LOCAL_MEM_FENCE);                               \
                in[lid/redsz] = in[lid]; \
            } \
            else \
            { \
                barrier(CLK_LOCAL_MEM_FENCE);                               \
                break;\
            } \
        } while ((remain = remain/redsz) >= redsz); \
        out[get_group_id(1)*get_num_groups(0) + get_group_id(0)] = in[0]; \
        /*if (0 == lid)                                               \
        {                                                           \
            for (size_t offset = 1; offset < BLOCK_SZ/vecsz; offset++)    \
            {                                                       \
                in[0] = operation(in[0], in[offset]);               \
            }                                                       \
            out[get_group_id(1)*get_num_groups(0) + get_group_id(0)] = in[0]; \
        }*/ \
        /*for (size_t offset = BLOCK_SZ / (2*vecsz); offset > 0; offset /= 2) \
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
            out[get_group_id(1)*get_num_groups(0) + get_group_id(0)] = in[0]; \
        }*/ \
    }
    #endif
#endif

#else

    #error No device type specified - don't know which reduction to use

#endif

