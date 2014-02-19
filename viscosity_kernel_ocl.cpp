#include "ocl_common.hpp"
extern CloverChunk chunk;

extern "C" void viscosity_kernel_ocl_
(int *xmin,int *x_max,int *ymin,int *y_max,
const double *celldx,
const double *celldy,
const double *density0,
const double *pressure,
double *viscosity,
const double *xvel0,
const double *yvel0)
{
    chunk.viscosity_kernel();
}

void CloverChunk::viscosity_kernel
(void)
{
    //ENQUEUE(viscosity_device)
    ENQUEUE_OFFSET(viscosity_device)
}

