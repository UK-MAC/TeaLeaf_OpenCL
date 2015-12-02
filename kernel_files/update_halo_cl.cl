#include <kernel_files/macros_cl.cl>

__kernel void update_halo_left
(kernel_info_t kernel_info,
cell_info_t grid_type,
__GLOBAL__ double * __restrict const cur_array,
int depth)
{
    int x_extra = grid_type.x_extra;
    int y_extra = grid_type.y_extra;
    int x_invert = grid_type.x_invert;
    int y_invert = grid_type.y_invert;
    int x_face = grid_type.x_face;
    int y_face = grid_type.y_face;

    __kernel_indexes;

    if (row >= HALO_DEPTH - depth && row <= (y_max + HALO_DEPTH - 1) + y_extra + depth)
    {
        const int src = 1 + (HALO_DEPTH - column - 1)*2;
        const int dst = 0;
        cur_array[THARR2D(dst, 0, x_extra)] = y_invert * cur_array[THARR2D(src, 0, x_extra)];
    }
}

__kernel void update_halo_right
(kernel_info_t kernel_info,
cell_info_t grid_type,
__GLOBAL__ double * __restrict const cur_array,
int depth)
{
    int x_extra = grid_type.x_extra;
    int y_extra = grid_type.y_extra;
    int x_invert = grid_type.x_invert;
    int y_invert = grid_type.y_invert;
    int x_face = grid_type.x_face;
    int y_face = grid_type.y_face;

    __kernel_indexes;

    if (row >= HALO_DEPTH - depth && row <= (y_max + HALO_DEPTH - 1) + y_extra + depth)
    {
        const int src = x_max + x_extra;
        const int dst = x_max + x_extra + (HALO_DEPTH - column - 1)*2 + 1;
        cur_array[THARR2D(dst, 0, x_extra)] = cur_array[THARR2D(src, 0, x_extra)];
    }
}

__kernel void update_halo_bottom
(kernel_info_t kernel_info,
cell_info_t grid_type,
__GLOBAL__ double * __restrict const cur_array,
int depth)
{
    int x_extra = grid_type.x_extra;
    int y_extra = grid_type.y_extra;
    int x_invert = grid_type.x_invert;
    int y_invert = grid_type.y_invert;
    int x_face = grid_type.x_face;
    int y_face = grid_type.y_face;

    __kernel_indexes;

    if (column >= HALO_DEPTH - depth && column <= (x_max + HALO_DEPTH - 1) + x_extra + depth)
    {
        const int src = 1 + (HALO_DEPTH - row - 1)*2;
        const int dst = 0;
        cur_array[THARR2D(0, dst, x_extra)] = y_invert * cur_array[THARR2D(0, src, x_extra)];
    }
}

__kernel void update_halo_top
(kernel_info_t kernel_info,
cell_info_t grid_type,
__GLOBAL__ double * __restrict const cur_array,
int depth)
{
    int x_extra = grid_type.x_extra;
    int y_extra = grid_type.y_extra;
    int x_invert = grid_type.x_invert;
    int y_invert = grid_type.y_invert;
    int x_face = grid_type.x_face;
    int y_face = grid_type.y_face;

    __kernel_indexes;

    if (column >= HALO_DEPTH - depth && column <= (x_max + HALO_DEPTH - 1) + x_extra + depth)
    {
        const int src = y_max + y_extra;
        const int dst = y_max + y_extra + (HALO_DEPTH - row - 1)*2 + 1;
        cur_array[THARR2D(0, dst, x_extra)] = cur_array[THARR2D(0, src, x_extra)];
    }
}

