#include <kernel_files/macros_cl.cl>
#include <kernel_files/tea_block_jacobi.cl>

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
(__global const double * const r1,
 __global const double * const r2,
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

        /*
         *  This is how the Fortran does it - makes no difference seeing as u/u0
         *  are set to 0 outside of the bounds of the mesh anyway, but this is
         *  more consistent and possibly prevent future bugs
         */
        if (row < (y_min + HALO_DEPTH))
        {
            dens_up=0;
        }
        if (column < (x_min + HALO_DEPTH))
        {
            dens_left=0;
        }

        Kx[THARR2D(0, 0, 0)] = (dens_left + dens_centre)/(2.0*dens_left*dens_centre);
        Kx[THARR2D(0, 0, 0)] *= rx;
        Ky[THARR2D(0, 0, 0)] = (dens_up + dens_centre)/(2.0*dens_up*dens_centre);
        Ky[THARR2D(0, 0, 0)] *= ry;

        // only inside bounds of mesh
        if (row <= (y_max + HALO_DEPTH) && column <= (x_max + HALO_DEPTH))
        {
            u0[THARR2D(0, 0, 0)] = energy[THARR2D(0, 0, 0)]*density[THARR2D(0, 0, 0)];
            u[THARR2D(0, 0, 0)] = energy[THARR2D(0, 0, 0)]*density[THARR2D(0, 0, 0)];
        }
    }
}

__kernel void tea_leaf_init_jac_diag
(__global       double * __restrict const Mi,
 __global const double * __restrict const Kx,
 __global const double * __restrict const Ky)
{
    __kernel_indexes;

    if (WITHIN_BOUNDS)
    {
        const double diag = (1.0
            + (Ky[THARR2D(0, 1, 0)] + Ky[THARR2D(0, 0, 0)])
            + (Kx[THARR2D(1, 0, 0)] + Kx[THARR2D(0, 0, 0)]));

        Mi[THARR2D(0, 0, 0)] = 1.0/diag;
    }
}

