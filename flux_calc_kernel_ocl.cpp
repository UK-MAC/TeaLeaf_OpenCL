
#include "ocl_common.hpp"
extern CloverChunk chunk;

extern "C" void flux_calc_kernel_ocl_
(int *xmin,int *xmax,int *ymin,int *ymax,
double *dbyt,
const double *xarea,
const double *yarea,
const double *xvel0,
const double *yvel0,
const double *xvel1,
const double *yvel1,
double *vol_flux_x,
double *vol_flux_y)
{
    chunk.flux_calc_kernel(*dbyt);
}

void CloverChunk::flux_calc_kernel
(double dbyt)
{
    flux_calc_device.setArg(0, dbyt);

    //ENQUEUE(flux_calc_device)
    ENQUEUE_OFFSET(flux_calc_device)
}

