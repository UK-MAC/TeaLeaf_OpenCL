#include <kernel_files/macros_cl.cl>

__kernel void update_halo_bottom
(int x_extra,   int y_extra,
 int x_invert,  int y_invert,
 int x_face,    int y_face,
 int grid_type, int depth, 
 __global double * __restrict const cur_array)
{
    __kernel_indexes;

    // offset by 1 if it is anything but a CELL grid
    int b_offset = (grid_type != CELL_DATA) ? 1 : 0;

    if (column >= HALO_DEPTH - depth && column <= (x_max + HALO_DEPTH) + x_extra + depth)
    {
        cur_array[THARR2D(0, 0, x_extra)] = y_invert * cur_array[THARR2D(0, (HALO_DEPTH - row)*2, x_extra)];
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

    // if x face data, offset source/dest by - 1
    int x_f_offset = (x_face) ? 1 : 0;

    if (column >= HALO_DEPTH - depth && column <= (x_max + HALO_DEPTH) + x_extra + depth)
    {
        cur_array[THARR2D(0, y_max + y_extra + HALO_DEPTH, x_extra)] = cur_array[THARR2D(0, y_max + y_extra + HALO_DEPTH - row*2, x_extra)];
    }
}

__kernel void update_halo_left
(int x_extra, int y_extra,
 int x_invert, int y_invert,
 int x_face, int y_face,
 int grid_type, int depth, 
 __global double * __restrict const cur_array)
{
    __kernel_indexes;

    if (row >= HALO_DEPTH - depth && row <= (y_max + HALO_DEPTH) + y_extra + depth)
    {
        cur_array[THARR2D(0, 0, x_extra)] = cur_array[THARR2D((HALO_DEPTH - column)*2, 0, x_extra)];
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

    if (row >= HALO_DEPTH - depth && row <= (y_max + HALO_DEPTH) + y_extra + depth)
    {
        cur_array[THARR2D(x_max + x_extra + HALO_DEPTH, 0, x_extra)] = cur_array[THARR2D(x_max + x_extra + HALO_DEPTH - column*2, 0, x_extra)];
    }
}

