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
        cur_array[THARR2D(0, 0, x_extra)] = cur_array[THARR2D((HALO_DEPTH - column)*2 - 1, 0, x_extra)];
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
        cur_array[THARR2D(x_max + x_extra + HALO_DEPTH - 1, 0, x_extra)] = cur_array[THARR2D(x_max + x_extra + HALO_DEPTH - column*2 - 1, 0, x_extra)];
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
        cur_array[THARR2D(0, 0, x_extra)] = y_invert * cur_array[THARR2D(0, (HALO_DEPTH - row)*2 - 1, x_extra)];
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
        cur_array[THARR2D(0, y_max + y_extra + HALO_DEPTH - 1, x_extra)] = cur_array[THARR2D(0, y_max + y_extra + HALO_DEPTH - row*2 - 1, x_extra)];
    }
}

