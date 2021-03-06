#include <kernel_files/macros_cl.cl>

__kernel void update_halo_left
(int x_extra, int y_extra,
 int x_invert, int y_invert,
 int x_face, int y_face,
 int grid_type, int depth,
 __global double * __restrict const cur_array)
{
    __kernel_indexes;

    if (row >= HALO_DEPTH - depth && row <= (y_max + HALO_DEPTH - 1) + y_extra + depth)
    {
        const size_t src = 1 + (HALO_DEPTH - column - 1)*2;
        const size_t dst = 0;
        cur_array[THARR2D(dst, 0, x_extra)] = y_invert * cur_array[THARR2D(src, 0, x_extra)];
    }
}

__kernel void update_halo_right
(int x_extra, int y_extra,
 int x_invert, int y_invert,
 int x_face, int y_face,
 int grid_type, int depth,
 __global double * __restrict const cur_array)
{
    __kernel_indexes;

    if (row >= HALO_DEPTH - depth && row <= (y_max + HALO_DEPTH - 1) + y_extra + depth)
    {
        const size_t src = x_max + x_extra;
        const size_t dst = x_max + x_extra + (HALO_DEPTH - column - 1)*2 + 1;
        cur_array[THARR2D(dst, 0, x_extra)] = cur_array[THARR2D(src, 0, x_extra)];
    }
}

__kernel void update_halo_bottom
(int x_extra,   int y_extra,
 int x_invert,  int y_invert,
 int x_face,    int y_face,
 int grid_type, int depth,
 __global double * __restrict const cur_array)
{
    __kernel_indexes;

    if (column >= HALO_DEPTH - depth && column <= (x_max + HALO_DEPTH - 1) + x_extra + depth)
    {
        const size_t src = 1 + (HALO_DEPTH - row - 1)*2;
        const size_t dst = 0;
        cur_array[THARR2D(0, dst, x_extra)] = y_invert * cur_array[THARR2D(0, src, x_extra)];
    }
}

__kernel void update_halo_top
(int x_extra, int y_extra,
 int x_invert, int y_invert,
 int x_face, int y_face,
 int grid_type, int depth,
 __global double * __restrict const cur_array)
{
    __kernel_indexes;

    if (column >= HALO_DEPTH - depth && column <= (x_max + HALO_DEPTH - 1) + x_extra + depth)
    {
        const size_t src = y_max + y_extra;
        const size_t dst = y_max + y_extra + (HALO_DEPTH - row - 1)*2 + 1;
        cur_array[THARR2D(0, dst, x_extra)] = cur_array[THARR2D(0, src, x_extra)];
    }
}

