
__kernel void generate_chunk
(__global const double * __restrict const vertexx,
 __global const double * __restrict const vertexy,
 __global const double * __restrict const cellx,
 __global const double * __restrict const celly,
 __global       double * __restrict const density0,
 __global       double * __restrict const energy0,
 __global       double * __restrict const xvel0,
 __global       double * __restrict const yvel0,

 __global const double * __restrict const state_density,
 __global const double * __restrict const state_energy,
 __global const double * __restrict const state_xvel,
 __global const double * __restrict const state_yvel,
 __global const double * __restrict const state_xmin,
 __global const double * __restrict const state_xmax,
 __global const double * __restrict const state_ymin,
 __global const double * __restrict const state_ymax,
 __global const double * __restrict const state_radius,
 __global const int    * __restrict const state_geometry,

 const int g_rect,
 const int g_circ,
 const int g_point,
 const int state)
{
    __kernel_indexes;

    IF_ROW_WITHIN(- 2+0, + 2+0)
    IF_COLUMN_WITHIN(- 2+0, + 2+0)
    {
        if(g_rect == state_geometry[state])
        {
            if(vertexx[1 + column] >= state_xmin[state]
            && vertexx[column] <  state_xmax[state]
            && vertexy[1 + row]    >= state_ymin[state]
            && vertexy[row]    <  state_ymax[state])
            {
                energy0[THARR2D(0, 0, 0)] = state_energy[state];
                density0[THARR2D(0, 0, 0)] = state_density[state];

                //unrolled do loop
                xvel0[THARR2D(0, 0, 1)] = state_xvel[state];
                yvel0[THARR2D(0, 0, 1)] = state_yvel[state];

                xvel0[THARR2D(1, 0, 1)] = state_xvel[state];
                yvel0[THARR2D(1, 0, 1)] = state_yvel[state];

                xvel0[THARR2D(0, 1, 1)] = state_xvel[state];
                yvel0[THARR2D(0, 1, 1)] = state_yvel[state];

                xvel0[THARR2D(1, 1, 1)] = state_xvel[state];
                yvel0[THARR2D(1, 1, 1)] = state_yvel[state];
            }
        }
        else if(state_geometry[state] == g_circ)
        {
            double radius = SQRT(cellx[column] * cellx[column] + celly[row] + celly[row]);
            if(radius <= state_radius[state])
            {
                energy0[THARR2D(0, 0, 0)] = state_energy[state];
                density0[THARR2D(0, 0, 0)] = state_density[state];

                //unrolled do loop
                xvel0[THARR2D(0, 0, 1)] = state_xvel[state];
                yvel0[THARR2D(0, 0, 1)] = state_yvel[state];

                xvel0[THARR2D(1, 0, 1)] = state_xvel[state];
                yvel0[THARR2D(1, 0, 1)] = state_yvel[state];

                xvel0[THARR2D(0, 1, 1)] = state_xvel[state];
                yvel0[THARR2D(0, 1, 1)] = state_yvel[state];

                xvel0[THARR2D(1, 1, 1)] = state_xvel[state];
                yvel0[THARR2D(1, 1, 1)] = state_yvel[state];
            }
        }
    }
}

__kernel void generate_chunk_init
(__global       double * density0,
 __global       double * energy0,
 __global       double * xvel0,
 __global       double * yvel0,
 __global const double * state_density,
 __global const double * state_energy,
 __global const double * state_xvel,
 __global const double * state_yvel)
{
    __kernel_indexes;

    IF_ROW_WITHIN(- 2+0, + 2+0)
    IF_COLUMN_WITHIN(- 2+0, + 2+0)
    {
        energy0[THARR2D(0, 0, 0)] = state_energy[0];
        density0[THARR2D(0, 0, 0)] = state_density[0];
        xvel0[THARR2D(0, 0, 1)] = state_xvel[0];
        yvel0[THARR2D(0, 0, 1)] = state_yvel[0];
    }
}

