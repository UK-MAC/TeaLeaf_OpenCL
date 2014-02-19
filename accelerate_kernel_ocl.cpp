#include "ocl_common.hpp"
extern CloverChunk chunk;

extern "C" void accelerate_kernel_ocl_
(int *xmin,int *xmax,int *ymin,int *ymax,
double *dbyt,
double *xarea,double *yarea,
double *volume,
double *density0,
double *pressure,
double *viscosity,
double *xvel0,
double *yvel0,
double *xvel1,
double *yvel1,
double *unused_array)
{
    chunk.accelerate_kernel(*dbyt);
}

void CloverChunk::accelerate_kernel
(double dbyt)
{
    accelerate_device.setArg(0, dbyt);

    //ENQUEUE(accelerate_device)
    ENQUEUE_OFFSET(accelerate_device)
}

