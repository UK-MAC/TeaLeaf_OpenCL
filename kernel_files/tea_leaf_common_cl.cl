#include <kernel_files/macros_cl.cl>
#include <kernel_files/tea_block_jacobi.cl>

__kernel void tea_leaf_finalise
(kernel_info_t kernel_info,
 __GLOBAL__ const double * __restrict const density,
 __GLOBAL__ const double * __restrict const u1,
 __GLOBAL__       double * __restrict const energy)
{
    __kernel_indexes;

    if (WITHIN_BOUNDS)
    {
        energy[THARR2D(0, 0, 0)] = u1[THARR2D(0, 0, 0)]/density[THARR2D(0, 0, 0)];
    }
}

__kernel void tea_leaf_calc_residual
(kernel_info_t kernel_info,
 __GLOBAL__ const double * __restrict const u,
 __GLOBAL__ const double * __restrict const u0,
 __GLOBAL__       double * __restrict const r,
 __GLOBAL__ const double * __restrict const Kx,
 __GLOBAL__ const double * __restrict const Ky)
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
(kernel_info_t kernel_info,
 __GLOBAL__ const double * const r1,
 __GLOBAL__ const double * const r2,
 __GLOBAL__       double * __restrict const rro)
{
    __kernel_indexes;

    __SHARED__ double rro_shared[BLOCK_SZ];
    rro_shared[lid] = 0.0;

    if (WITHIN_BOUNDS)
    {
        rro_shared[lid] = r1[THARR2D(0, 0, 0)]*r2[THARR2D(0, 0, 0)];
    }

    REDUCTION(rro_shared, rro, SUM)
}

__kernel void tea_leaf_init_common
(kernel_info_t kernel_info,
 __GLOBAL__ const double * __restrict const density,
 __GLOBAL__ const double * __restrict const energy,
 __GLOBAL__       double * __restrict const Kx,
 __GLOBAL__       double * __restrict const Ky,
 __GLOBAL__       double * __restrict const u0,
 __GLOBAL__       double * __restrict const u,
 double rx, double ry,
 const int coef)
{
    __kernel_indexes;

    if (WITHIN_BOUNDS)
    {
        double dens_centre, dens_left, dens_down;

        if (COEF_CONDUCTIVITY == coef)
        {
            dens_centre = density[THARR2D(0, 0, 0)];
            dens_left = density[THARR2D(-1, 0, 0)];
            dens_down = density[THARR2D(0, -1, 0)];
        }
        else if (COEF_RECIP_CONDUCTIVITY == coef)
        {
            dens_centre = 1.0/density[THARR2D(0, 0, 0)];
            dens_left = 1.0/density[THARR2D(-1, 0, 0)];
            dens_down = 1.0/density[THARR2D(0, -1, 0)];
        }

        Kx[THARR2D(0, 0, 0)] = (dens_left + dens_centre)/(2.0*dens_left*dens_centre);
        Kx[THARR2D(0, 0, 0)] *= rx;
        Ky[THARR2D(0, 0, 0)] = (dens_down + dens_centre)/(2.0*dens_down*dens_centre);
        Ky[THARR2D(0, 0, 0)] *= ry;
    }
}

__kernel void tea_leaf_zero_boundary
(kernel_info_t kernel_info,
 __GLOBAL__ double * __restrict const Kx,
 __GLOBAL__ double * __restrict const Ky,
 int zero_left, int zero_right, int zero_bottom, int zero_top)
{
    __kernel_indexes;

    if (WITHIN_BOUNDS)
    {
        if (zero_left && column <= HALO_DEPTH)
        {
            Kx[THARR2D(0, 0, 0)] = 0;
            if (column < HALO_DEPTH) Ky[THARR2D(0, 0, 0)] = 0;
        }

        if (zero_right && column > (x_max - 1) + HALO_DEPTH)
        {
            Kx[THARR2D(0, 0, 0)] = 0;
            Ky[THARR2D(0, 0, 0)] = 0;
        }

        if (zero_bottom && row <= HALO_DEPTH)
        {
            Ky[THARR2D(0, 0, 0)] = 0;
            if (row < HALO_DEPTH) Kx[THARR2D(0, 0, 0)] = 0;
        }

        if (zero_top && row > (y_max - 1) + HALO_DEPTH)
        {
            Ky[THARR2D(0, 0, 0)] = 0;
            Kx[THARR2D(0, 0, 0)] = 0;
        }
    }
}

__kernel void tea_leaf_init_jac_diag
(kernel_info_t kernel_info,
 __GLOBAL__       double * __restrict const Mi,
 __GLOBAL__ const double * __restrict const Kx,
 __GLOBAL__ const double * __restrict const Ky)
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

