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

    if (column >= 2 - depth && column <= (x_max + 1) + x_extra + depth)
    {
        const int offset = 2 + b_offset;

        /*
         * 1 - 2 * row means that row 0 services row 1, and vice versa
         * this means that it can be dispatched with 'depth' rows only
         */
        cur_array[THARR2D(0, 1 - (2 * row), x_extra)] = y_invert * cur_array[THARR2D(0, offset, x_extra)];
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

    if (column >= 2 - depth && column <= (x_max + 1) + x_extra + depth)
    {
        const int offset = (- row) * 2 - 1 - x_f_offset;

        cur_array[THARR2D(0, y_extra + y_max + 2, x_extra)] = y_invert * cur_array[THARR2D(0, y_max + 2 + offset, x_extra)];
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

    if (row >= 2 - depth && row <= (y_max + 1) + y_extra + depth)
    {
        cur_array[THARR2D(1 - 2*column, 0, x_extra)] = cur_array[THARR2D((x_min + 1) + x_extra, 0, x_extra)];
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

    if (row >= 2 - depth && row <= (y_max + 1) + y_extra + depth)
    {
        cur_array[THARR2D((x_max + 1) + x_extra + 1, 0, x_extra)] = cur_array[THARR2D((x_max + 1) - 2*column, 0, x_extra)];
    }
}

