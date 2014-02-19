__kernel void flux_calc
(double dt,
 __global const double * __restrict const xarea,
 __global const double * __restrict const yarea,
 __global const double * __restrict const xvel0,
 __global const double * __restrict const yvel0,
 __global const double * __restrict const xvel1,
 __global const double * __restrict const yvel1,
 __global       double * __restrict const vol_flux_x,
 __global       double * __restrict const vol_flux_y)
{
    __kernel_indexes;

    //IF_ROW_WITHIN(+0, +0)
    //IF_COLUMN_WITHIN(+0, + 1+0)
    {
        vol_flux_x[THARR2D(0, 0, 1)] = 0.25 * dt * xarea[THARR2D(0, 0, 1)]
            * (xvel0[THARR2D(0, 0, 1)] + xvel0[THARR2D(0, 1, 1)]
            + xvel1[THARR2D(0, 0, 1)] + xvel1[THARR2D(0, 1, 1)]);
    }

    //IF_ROW_WITHIN(+0, + 1+0)
    //IF_COLUMN_WITHIN(+0, +0)
    {
        vol_flux_y[THARR2D(0, 0, 0)] = 0.25 * dt * yarea[THARR2D(0, 0, 0)]
            * (yvel0[THARR2D(0, 0, 1)] + yvel0[THARR2D(1, 0, 1)]
            + yvel1[THARR2D(0, 0, 1)] + yvel1[THARR2D(1, 0, 1)]);
    }

}

