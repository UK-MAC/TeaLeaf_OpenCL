#include "ocl_common.hpp"
#include "ocl_reduction.hpp"
extern CloverChunk chunk;

extern "C" void field_summary_kernel_ocl_
(int *x_min,int *x_max,int *y_min,int *y_max,
const double* volume,
const double* density0,
const double* energy0,
const double* u,
const double* pressure,
const double* xvel0,
const double* yvel0,

double* vol,
double* mass,
double* ie,
double* ke,
double* press,
double* temp)
{
    chunk.field_summary_kernel(vol, mass, ie, ke, press, temp);
}

void CloverChunk::field_summary_kernel
(double* vol, double* mass, double* ie, double* ke, double* press, double* temp)
{
    ENQUEUE(field_summary_device);

    queue.finish();

    *vol = reduceValue<double>(sum_red_kernels_double, reduce_buf_1);
    *mass = reduceValue<double>(sum_red_kernels_double, reduce_buf_2);
    *ie = reduceValue<double>(sum_red_kernels_double, reduce_buf_3);
    *ke = reduceValue<double>(sum_red_kernels_double, reduce_buf_4);
    *press = reduceValue<double>(sum_red_kernels_double, reduce_buf_5);
    *temp = reduceValue<double>(sum_red_kernels_double, reduce_buf_6);
}

