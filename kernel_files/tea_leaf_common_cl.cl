#include <kernel_files/macros_cl.cl>

__kernel void tea_leaf_finalise
(__global const double * __restrict const density,
 __global const double * __restrict const u1,
 __global       double * __restrict const energy)
{
    __kernel_indexes;

    if (WITHIN_BOUNDS)
    {
        energy[THARR2D(0, 0, 0)] = u1[THARR2D(0, 0, 0)]/density[THARR2D(0, 0, 0)];
    }
}

__kernel void tea_leaf_calc_residual
(__global const double * __restrict const u,
 __global const double * __restrict const u0,
 __global       double * __restrict const r,
 __global const double * __restrict const Kx,
 __global const double * __restrict const Ky)
{
    __kernel_indexes;

    if (WITHIN_BOUNDS)
    {
        const double smvp = (1.0
            + (Ky[THARR2D(0, 1, 0)] + Ky[THARR2D(0, 0, 0)])
            + (Kx[THARR2D(1, 0, 0)] + Kx[THARR2D(0, 0, 0)]))*u[THARR2D(0, 0, 0)]
            - (Ky[THARR2D(0, 1, 0)]*u[THARR2D(0, 1, 0)] + Ky[THARR2D(0, 0, 0)]*u[THARR2D(0, -1, 0)])
            - (Kx[THARR2D(1, 0, 0)]*u[THARR2D(1, 0, 0)] + Kx[THARR2D(0, 0, 0)]*u[THARR2D(-1, 0, 0)]);

        r[THARR2D(0, 0, 0)] = u0[THARR2D(0, 0, 0)] - smvp;
    }
}

__kernel void tea_leaf_calc_2norm
(__global const double * __restrict const r1,
 __global const double * __restrict const r2,
 __global       double * __restrict const rro)
{
    __kernel_indexes;

    __local double rro_shared[BLOCK_SZ];
    rro_shared[lid] = 0.0;

    if (WITHIN_BOUNDS)
    {
        rro_shared[lid] = r1[THARR2D(0, 0, 0)]*r2[THARR2D(0, 0, 0)];
    }

    REDUCTION(rro_shared, rro, SUM)
}

#define COEF_CONDUCTIVITY 1
#define COEF_RECIP_CONDUCTIVITY 2
__kernel void tea_leaf_init_common
(__global const double * __restrict const density,
 __global const double * __restrict const energy,
 __global       double * __restrict const Kx,
 __global       double * __restrict const Ky,
 __global       double * __restrict const u0,
 __global       double * __restrict const u,
 double rx, double ry,
 const int coef)
{
    __kernel_indexes;

    if (WITHIN_BOUNDS)
    {
        u0[THARR2D(0, 0, 0)] = energy[THARR2D(0, 0, 0)]*density[THARR2D(0, 0, 0)];
        u[THARR2D(0, 0, 0)] = energy[THARR2D(0, 0, 0)]*density[THARR2D(0, 0, 0)];

        // don't do this bit in second row/column
        if (row >= (y_min + 1)
        && column >= (x_min + 1))
        {
            double dens_centre, dens_left, dens_up;

            if (COEF_CONDUCTIVITY == coef)
            {
                dens_centre = density[THARR2D(0, 0, 0)];
                dens_left = density[THARR2D(-1, 0, 0)];
                dens_up = density[THARR2D(0, -1, 0)];
            }
            else if (COEF_RECIP_CONDUCTIVITY == coef)
            {
                dens_centre = 1.0/density[THARR2D(0, 0, 0)];
                dens_left = 1.0/density[THARR2D(-1, 0, 0)];
                dens_up = 1.0/density[THARR2D(0, -1, 0)];
            }

            Kx[THARR2D(0, 0, 0)] = (dens_left + dens_centre)/(2.0*dens_left*dens_centre);
            Kx[THARR2D(0, 0, 0)] *= rx;
            Ky[THARR2D(0, 0, 0)] = (dens_up + dens_centre)/(2.0*dens_up*dens_centre);
            Ky[THARR2D(0, 0, 0)] *= ry;
        }
    }
}

#define COEF_A (1*(-Ky[THARR2D(0,j+ 0, 0)]))
#define COEF_B (1*(1.0 + (Ky[THARR2D(0,j+ 1, 0)] + Ky[THARR2D(0,j+ 0, 0)]) + (Kx[THARR2D(1,j+ 0, 0)] + Kx[THARR2D(0,j+ 0, 0)])))
#define COEF_C (1*(-Ky[THARR2D(0,j+ 1, 0)]))

__kernel void block_init
(__global const double * __restrict const r,
 __global const double * __restrict const z,
 __global       double * __restrict const cp,
 __global       double * __restrict const bfp,
 __global const double * __restrict const dp,
 __global const double * __restrict const Kx,
 __global const double * __restrict const Ky)
{
    const size_t column = get_global_id(0);
    const size_t row = get_global_id(1)*BLOCK_SIZE + 2;

    if (row > y_max || column > x_max) return;

    int j = 0;

    cp[THARR2D(0, j, 0)] = COEF_C/COEF_B;

    for (j = 1; j < BLOCK_SIZE; j++)
    {
        bfp[THARR2D(0, j, 0)] = 1.0/(COEF_B - COEF_A*cp[THARR2D(0, j - 1, 0)]);
        cp[THARR2D(0, j, 0)] = COEF_C*bfp[THARR2D(0, j, 0)];
    }
}

__kernel void block_solve
(__global const double * __restrict const r,
 __global       double * __restrict const z,
 __global const double * __restrict const cp,
 __global const double * __restrict const bfp,
 __global       double * __restrict const dp,
 __global const double * __restrict const Kx,
 __global const double * __restrict const Ky)
{
    const size_t column = get_global_id(0);
    const size_t row = get_global_id(1)*BLOCK_SIZE + 2;

    if (row > y_max || column > x_max) return;

    int j = 0;

    __private double dp_l[BLOCK_SIZE];

    dp_l[j] = r[THARR2D(0, j, 0)]/COEF_B;

    for (j = 1; j < BLOCK_SIZE; j++)
    {
        dp_l[j] = (r[THARR2D(0, j, 0)] - COEF_A*dp_l[j - 1])*bfp[THARR2D(0, j, 0)];
    }

    j = BLOCK_SIZE - 1;

    __private double z_l[BLOCK_SIZE];
    z_l[j] = dp_l[j];

    for (j = BLOCK_SIZE - 2; j >= 0; j--)
    {
        z_l[j] = dp_l[j] - cp[THARR2D(0, j, 0)]*z_l[j + 1];
    }

    for (j = 0; j < BLOCK_SIZE; j++)
    {
        z[THARR2D(0, j, 0)] = z_l[j];
    }
}

