#include "ocl_common.hpp"
#include "ocl_reduction.hpp"
extern CloverChunk chunk;

extern "C" void pdv_kernel_ocl_
(int *errorcondition,int *prdct,
int *xmin,int *xmax,int *ymin,int *ymax,double *dtbyt,
double *xarea,double *yarea,double *volume,
double *density0,
double *density1,
double *energy0,
double *energy1,
double *pressure,
double *viscosity,
double *xvel0,
double *xvel1,
double *yvel0,
double *yvel1,
double *unused_array)
{
    chunk.PdV_kernel(errorcondition, *prdct, *dtbyt);
}

void CloverChunk::PdV_kernel
(int* error_condition, int predict, double dt)
{
    if (1 == predict)
    {
        PdV_predict_device.setArg(0, dt);

        //ENQUEUE(PdV_predict_device)
        ENQUEUE_OFFSET(PdV_predict_device)
    }
    else
    {
        PdV_not_predict_device.setArg(0, dt);

        //ENQUEUE(PdV_not_predict_device)
        ENQUEUE_OFFSET(PdV_not_predict_device)
    }

    *error_condition = reduceValue<int>(max_red_kernels_int,
                                        PdV_reduce_buf);

    if (1 == *error_condition)
    {
        fprintf(stdout, "Negative volume in PdV kernel\n");
    }
    else if (2 == *error_condition)
    {
        fprintf(stdout, "Negative cell volume in PdV kernel\n");
    }
}

