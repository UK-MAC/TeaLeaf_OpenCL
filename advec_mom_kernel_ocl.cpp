#include "ocl_common.hpp"
extern CloverChunk chunk;

extern "C" void advec_mom_kernel_ocl_
(int *xmin,int *xmax,int *ymin,int *ymax,
      double *xvel1,
      double *yvel1,
const double *mass_flux_x,
const double *vol_flux_x,
const double *mass_flux_y,
const double *vol_flux_y,
const double *volume,
const double *density1,

double* unused_array1,
double* unused_array2,
double* unused_array3,
double* unused_array4,
double* unused_array5,
double* unused_array6,
double* unused_array7,

const double *celldx,
const double *celldy,

int *whch_vl,
int *swp_nmbr,
int *drctn,
int *vector)
{
    chunk.advec_mom_kernel(*whch_vl, *swp_nmbr, *drctn);
}

void CloverChunk::advec_mom_kernel
(int which_vel, int sweep_number, int direction)
{
    int mom_sweep = direction + (2 * (sweep_number - 1));

    advec_mom_vol_device.setArg(0, mom_sweep);

    //ENQUEUE(advec_mom_vol_device);
    ENQUEUE_OFFSET(advec_mom_vol_device);

    if (1 == which_vel)
    {
        advec_mom_flux_x_device.setArg(3, xvel1);
        advec_mom_xvel_device.setArg(3, xvel1);
        advec_mom_flux_y_device.setArg(3, xvel1);
        advec_mom_yvel_device.setArg(3, xvel1);
    }
    else
    {
        advec_mom_flux_x_device.setArg(3, yvel1);
        advec_mom_xvel_device.setArg(3, yvel1);
        advec_mom_flux_y_device.setArg(3, yvel1);
        advec_mom_yvel_device.setArg(3, yvel1);
    }

    if (1 == direction)
    {
        ENQUEUE(advec_mom_node_flux_post_x_device);
        ENQUEUE(advec_mom_node_pre_x_device);
        ENQUEUE(advec_mom_flux_x_device);
        ENQUEUE(advec_mom_xvel_device);
    }
    else if (2 == direction)
    {
        ENQUEUE(advec_mom_node_flux_post_y_device);
        ENQUEUE(advec_mom_node_pre_y_device);
        ENQUEUE(advec_mom_flux_y_device);
        ENQUEUE(advec_mom_yvel_device);
    }
}

