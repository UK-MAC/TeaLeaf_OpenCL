
__kernel void revert
(__global const double * __restrict const density0,
 __global       double * __restrict const density1,
 __global const double * __restrict const energy0,
 __global       double * __restrict const energy1)
{
    __kernel_indexes;

    //IF_ROW_WITHIN(+0, +0)
    //IF_COLUMN_WITHIN(+0, +0)
    {
        density1[THARR2D(0, 0, 0)] = density0[THARR2D(0, 0, 0)];
        energy1[THARR2D(0, 0, 0)] = energy0[THARR2D(0, 0, 0)];
    }
}

