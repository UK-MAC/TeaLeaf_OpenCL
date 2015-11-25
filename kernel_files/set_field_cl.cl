#include <kernel_files/macros_cl.cl>

__kernel void set_field
(kernel_info_t kernel_info,
 __GLOBAL__ const double* __restrict const energy0,
 __GLOBAL__       double* __restrict const energy1)
{
    __kernel_indexes;

    if (WITHIN_BOUNDS)
    {
        energy1[THARR2D(0, 0, 0)] = energy0[THARR2D(0, 0, 0)];
    }
}
