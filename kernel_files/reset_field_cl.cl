
__kernel void reset_field
(__global       double* __restrict const density0,
 __global const double* __restrict const density1,
 __global       double* __restrict const energy0,
 __global const double* __restrict const energy1,
 __global       double* __restrict const xvel0,
 __global const double* __restrict const xvel1,
 __global       double* __restrict const yvel0,
 __global const double* __restrict const yvel1)
{
    __kernel_indexes;

    //IF_ROW_WITHIN(+0, + 1+0)
    //IF_COLUMN_WITHIN(+0, + 1+0)
    {
        xvel0[THARR2D(0, 0, 1)] = xvel1[THARR2D(0, 0, 1)];
        yvel0[THARR2D(0, 0, 1)] = yvel1[THARR2D(0, 0, 1)];

        //if (row <= (y_max + 1)
        //&& column <= (x_max + 1))
        {
            density0[THARR2D(0, 0, 0)] = density1[THARR2D(0, 0, 0)];
            energy0[THARR2D(0, 0, 0)]  = energy1[THARR2D(0, 0, 0)];
        }
    }
}

