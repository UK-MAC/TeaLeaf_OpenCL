#include "./kernel_files/macros_cl.cl"

// for left/right
#define VERT_IDX                        \
    (column - get_global_offset(0)) +   \
    (row -    get_global_offset(1))*depth + offset

// for top/bottom
#define HORZ_IDX                                        \
    (column - get_global_offset(0)) +             \
    (row -    get_global_offset(1))*(x_max + x_extra + 2*depth) + offset

__kernel void pack_left_buffer
(kernel_info_t kernel_info,
 int x_extra, int y_extra,
 const  __GLOBAL__ double * __restrict cur_array,
        __GLOBAL__ double * __restrict left_buffer,
 const int depth, int offset)
{
    __kernel_indexes;

    if (row >= HALO_DEPTH - depth && row <= (y_max + HALO_DEPTH - 1) + y_extra + depth)
    {
        const int src = 1 + (HALO_DEPTH - column - 1)*2;
        left_buffer[VERT_IDX] = cur_array[THARR2D(src, 0, x_extra)];
    }
}

__kernel void unpack_left_buffer
(kernel_info_t kernel_info,
 int x_extra, int y_extra,
        __GLOBAL__ double * __restrict cur_array,
 const  __GLOBAL__ double * __restrict left_buffer,
 const int depth, int offset)
{
    __kernel_indexes;

    if (row >= HALO_DEPTH - depth && row <= (y_max + HALO_DEPTH - 1) + y_extra + depth)
    {
        const int dst = 0;
        cur_array[THARR2D(dst, 0, x_extra)] = left_buffer[VERT_IDX];
    }
}

/************************************************************/

__kernel void pack_right_buffer
(kernel_info_t kernel_info,
 int x_extra, int y_extra,
 const  __GLOBAL__ double * __restrict cur_array,
        __GLOBAL__ double * __restrict right_buffer,
 const int depth, int offset)
{
    __kernel_indexes;

    if (row >= HALO_DEPTH - depth && row <= (y_max + HALO_DEPTH - 1) + y_extra + depth)
    {
        const int src = x_max + x_extra;
        right_buffer[VERT_IDX] = cur_array[THARR2D(src, 0, x_extra)];
    }
}

__kernel void unpack_right_buffer
(kernel_info_t kernel_info,
 int x_extra, int y_extra,
        __GLOBAL__ double * __restrict cur_array,
 const  __GLOBAL__ double * __restrict right_buffer,
 const int depth, int offset)
{
    __kernel_indexes;

    if (row >= HALO_DEPTH - depth && row <= (y_max + HALO_DEPTH - 1) + y_extra + depth)
    {
        const int dst = x_max + x_extra + (HALO_DEPTH - column - 1)*2 + 1;
        cur_array[THARR2D(dst, 0, x_extra)] = right_buffer[VERT_IDX];
    }
}

/************************************************************/

__kernel void pack_bottom_buffer
(kernel_info_t kernel_info,
 int x_extra, int y_extra,
  __GLOBAL__ double * __restrict cur_array,
  __GLOBAL__ double * __restrict bottom_buffer,
 const int depth, int offset)
{
    __kernel_indexes;

    if (column >= HALO_DEPTH - depth && column <= (x_max + HALO_DEPTH - 1) + x_extra + depth)
    {
        const int src = 1 + (HALO_DEPTH - row - 1)*2;
        bottom_buffer[HORZ_IDX] = cur_array[THARR2D(0, src, x_extra)];
    }
}

__kernel void unpack_bottom_buffer
(kernel_info_t kernel_info,
 int x_extra, int y_extra,
  __GLOBAL__ double * __restrict cur_array,
  __GLOBAL__ double * __restrict bottom_buffer,
 const int depth, int offset)
{
    __kernel_indexes;

    if (column >= HALO_DEPTH - depth && column <= (x_max + HALO_DEPTH - 1) + x_extra + depth)
    {
        const int dst = 0;
        cur_array[THARR2D(dst, 0, x_extra)] = bottom_buffer[HORZ_IDX];
    }
}

/************************************************************/

__kernel void pack_top_buffer
(kernel_info_t kernel_info,
 int x_extra, int y_extra,
  __GLOBAL__ double * __restrict cur_array,
  __GLOBAL__ double * __restrict top_buffer,
 const int depth, int offset)
{
    __kernel_indexes;

    if (column >= HALO_DEPTH - depth && column <= (x_max + HALO_DEPTH - 1) + x_extra + depth)
    {
        const int src = y_max + y_extra;
        top_buffer[HORZ_IDX] = cur_array[THARR2D(0, src, x_extra)];
    }
}

__kernel void unpack_top_buffer
(kernel_info_t kernel_info,
 int x_extra, int y_extra,
  __GLOBAL__ double * __restrict cur_array,
  __GLOBAL__ double * __restrict top_buffer,
 const int depth, int offset)
{
    __kernel_indexes;

    if (column >= HALO_DEPTH - depth && column <= (x_max + HALO_DEPTH - 1) + x_extra + depth)
    {
        const int dst = y_max + y_extra + (HALO_DEPTH - row - 1)*2 + 1;
        cur_array[THARR2D(0, dst, x_extra)] = top_buffer[HORZ_IDX];
    }
}

