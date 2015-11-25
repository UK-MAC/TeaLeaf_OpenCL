#include <kernel_files/macros_cl.cl>

__kernel void generate_chunk
(kernel_info_t kernel_info,
 __GLOBAL__ const double * __restrict const vertexx,
 __GLOBAL__ const double * __restrict const vertexy,
 __GLOBAL__ const double * __restrict const cellx,
 __GLOBAL__ const double * __restrict const celly,
 __GLOBAL__       double * __restrict const density,
 __GLOBAL__       double * __restrict const energy0,

 __GLOBAL__ const double * __restrict const state_density,
 __GLOBAL__ const double * __restrict const state_energy,
 __GLOBAL__ const double * __restrict const state_xmin,
 __GLOBAL__ const double * __restrict const state_xmax,
 __GLOBAL__ const double * __restrict const state_ymin,
 __GLOBAL__ const double * __restrict const state_ymax,
 __GLOBAL__ const double * __restrict const state_radius,
 __GLOBAL__ const int    * __restrict const state_geometry,

 const int g_rect,
 const int g_circ,
 const int g_point,

 const int state)
{
    __kernel_indexes;

    if (WITHIN_BOUNDS)
    {
        const double x_cent = state_xmin[state];
        const double y_cent = state_ymin[state];

        if (g_rect == state_geometry[state])
        {
            if (vertexx[1 + column] >= state_xmin[state]
            && vertexx[column] <  state_xmax[state]
            && vertexy[1 + row]    >= state_ymin[state]
            && vertexy[row]    <  state_ymax[state])
            {
                energy0[THARR2D(0, 0, 0)] = state_energy[state];
                density[THARR2D(0, 0, 0)] = state_density[state];
            }
        }
        else if (state_geometry[state] == g_circ)
        {
            double x_pos = cellx[column]-x_cent;
            double y_pos = celly[row]-y_cent;
            double radius = SQRT(x_pos*x_pos + y_pos*y_pos);

            if (radius <= state_radius[state])
            {
                energy0[THARR2D(0, 0, 0)] = state_energy[state];
                density[THARR2D(0, 0, 0)] = state_density[state];
            }
        }
        else if (state_geometry[state] == g_point)
        {
            if (vertexx[column] == x_cent && vertexy[row] == y_cent)
            {
                energy0[THARR2D(0, 0, 0)] = state_energy[state];
                density[THARR2D(0, 0, 0)] = state_density[state];
            }
        }
    }
}

__kernel void generate_chunk_init_u
(kernel_info_t kernel_info,
 __GLOBAL__ const double * density,
 __GLOBAL__ const double * energy,
 __GLOBAL__       double * u,
 __GLOBAL__       double * u0)
{
    __kernel_indexes;

    if (WITHIN_BOUNDS)
    {
        u[THARR2D(0, 0, 0)] = energy[THARR2D(0, 0, 0)]*density[THARR2D(0, 0, 0)];
        u0[THARR2D(0, 0, 0)] = energy[THARR2D(0, 0, 0)]*density[THARR2D(0, 0, 0)];
    }
}

__kernel void generate_chunk_init
(kernel_info_t kernel_info,
 __GLOBAL__       double * density,
 __GLOBAL__       double * energy0,
 __GLOBAL__ const double * state_density,
 __GLOBAL__ const double * state_energy)
{
    __kernel_indexes;

    if (WITHIN_BOUNDS)
    {
        energy0[THARR2D(0, 0, 0)] = state_energy[0];
        density[THARR2D(0, 0, 0)] = state_density[0];
    }
}

