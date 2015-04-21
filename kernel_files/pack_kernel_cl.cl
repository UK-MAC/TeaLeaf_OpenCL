#include "./kernel_files/macros_cl.cl"

/********************/

#if 1
    // for left/right
    #define VERT_IDX                        \
        (((column - get_global_offset(0)) - 1) +                     \
        (((row -    get_global_offset(1)) - 1) + depth - 1)*depth)+offset+1
    // for top/bottom
    #define HORZ_IDX                        \
        (((column - get_global_offset(0)) - 1) + depth +             \
        (((row -    get_global_offset(1)) - 0) - 0)*(x_max + x_extra + 2*depth))+offset-1
#else
    #define HORZ_IDX \
        offset + column + (row + depth - 1)*depth - 2
    #define VERT_IDX gid+offset
#endif

__kernel void pack_left_buffer
(int x_extra, int y_extra,
const  __global double * __restrict cur_array,
       __global double * __restrict left_buffer,
const int depth, int offset)
{
    __kernel_indexes;

    if (row >= HALO_DEPTH - depth && row <= (y_max + HALO_DEPTH - 1) + y_extra + depth)
    {
        left_buffer[VERT_IDX] = cur_array[THARR2D((HALO_DEPTH - column)*2 - 1, 0, x_extra)];
    }
}

__kernel void unpack_left_buffer
(int x_extra, int y_extra,
       __global double * __restrict cur_array,
const  __global double * __restrict left_buffer,
const int depth, int offset)
{
    __kernel_indexes;

    if (row >= HALO_DEPTH - depth && row <= (y_max + HALO_DEPTH - 1) + y_extra + depth)
    {
        cur_array[THARR2D(0, 0, x_extra)] = left_buffer[VERT_IDX];
    }
}

/************************************************************/

__kernel void pack_right_buffer
(int x_extra, int y_extra,
const  __global double * __restrict cur_array,
       __global double * __restrict right_buffer,
const int depth, int offset)
{
    __kernel_indexes;

    if (row >= HALO_DEPTH - depth && row <= (y_max + HALO_DEPTH - 1) + y_extra + depth)
    {
        right_buffer[VERT_IDX] = cur_array[THARR2D(x_max + x_extra + HALO_DEPTH - column*2 - 1, 0, x_extra)];
    }
}

__kernel void unpack_right_buffer
(int x_extra, int y_extra,
       __global double * __restrict cur_array,
const  __global double * __restrict right_buffer,
const int depth, int offset)
{
    __kernel_indexes;

    if (row >= HALO_DEPTH - depth && row <= (y_max + HALO_DEPTH - 1) + y_extra + depth)
    {
        cur_array[THARR2D(x_max + x_extra + HALO_DEPTH, 0, x_extra)] = right_buffer[VERT_IDX];
    }
}

/************************************************************/

__kernel void pack_bottom_buffer
(int x_extra, int y_extra,
 __global double * __restrict cur_array,
 __global double * __restrict bottom_buffer,
const int depth, int offset)
{
    __kernel_indexes;

    if (column >= HALO_DEPTH - depth && column <= (x_max + HALO_DEPTH - 1) + x_extra + depth)
    {
        bottom_buffer[HORZ_IDX] = cur_array[THARR2D(0, (HALO_DEPTH - row)*2 - 1, x_extra)];
    }
}

__kernel void unpack_bottom_buffer
(int x_extra, int y_extra,
 __global double * __restrict cur_array,
 __global double * __restrict bottom_buffer,
const int depth, int offset)
{
    __kernel_indexes;

    if (column >= HALO_DEPTH - depth && column <= (x_max + HALO_DEPTH - 1) + x_extra + depth)
    {
        cur_array[THARR2D(0, 0, x_extra)] = bottom_buffer[HORZ_IDX];
    }
}

/************************************************************/

__kernel void pack_top_buffer
(int x_extra, int y_extra,
 __global double * __restrict cur_array,
 __global double * __restrict top_buffer,
const int depth, int offset)
{
    __kernel_indexes;

    if (column >= HALO_DEPTH - depth && column <= (x_max + HALO_DEPTH - 1) + x_extra + depth)
    {
        top_buffer[HORZ_IDX] = cur_array[THARR2D(0, y_max + y_extra + HALO_DEPTH - row*2 - 1, x_extra)];
    }
}

__kernel void unpack_top_buffer
(int x_extra, int y_extra,
 __global double * __restrict cur_array,
 __global double * __restrict top_buffer,
const int depth, int offset)
{
    __kernel_indexes;

    if (column >= HALO_DEPTH - depth && column <= (x_max + HALO_DEPTH - 1) + x_extra + depth)
    {
        cur_array[THARR2D(0, y_max + y_extra + HALO_DEPTH, x_extra)] = top_buffer[HORZ_IDX];
    }
}

