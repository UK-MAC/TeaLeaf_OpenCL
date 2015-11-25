#include <kernel_files/macros_cl.cl>

__kernel void field_summary
(kernel_info_t kernel_info,
 __GLOBAL__ const double * __restrict const volume,
 __GLOBAL__ const double * __restrict const density,
 __GLOBAL__ const double * __restrict const energy0,
 __GLOBAL__ const double * __restrict const u,

 __GLOBAL__       double * __restrict const vol,
 __GLOBAL__       double * __restrict const mass,
 __GLOBAL__       double * __restrict const ie,
 __GLOBAL__       double * __restrict const temp)
{
    __kernel_indexes;

    __SHARED__ double vol_shared[BLOCK_SZ];
    __SHARED__ double mass_shared[BLOCK_SZ];
    __SHARED__ double ie_shared[BLOCK_SZ];
    __SHARED__ double temp_shared[BLOCK_SZ];

    if (WITHIN_BOUNDS)
    {
        const double cell_vol = volume[THARR2D(0, 0, 0)];
        const double cell_mass = cell_vol * density[THARR2D(0, 0, 0)];

        vol_shared[lid] = cell_vol;
        mass_shared[lid] = cell_mass;
        ie_shared[lid] = cell_mass * energy0[THARR2D(0, 0, 0)];
        temp_shared[lid] = cell_mass*u[THARR2D(0, 0, 0)];
    }
    else
    {
        vol_shared[lid] = 0.0;
        mass_shared[lid] = 0.0;
        ie_shared[lid] = 0.0;
        temp_shared[lid] = 0.0;
    }

    REDUCTION(vol_shared, vol, SUM)
    REDUCTION(mass_shared, mass, SUM)
    REDUCTION(ie_shared, ie, SUM)
    REDUCTION(temp_shared, temp, SUM)
}

