#include "ocl_common.hpp"
extern CloverChunk chunk;

extern "C" void reset_field_kernel_ocl_
(int *x_min,int *x_max,int *y_min,int *y_max,
      double* density0,
const double* density1,
      double* energy0,
const double* energy1,
      double* xvel0,
const double* xvel1,
      double* yvel0,
const double* yvel1)
{
    chunk.reset_field_kernel();
}

void CloverChunk::reset_field_kernel
(void)
{
    //ENQUEUE(reset_field_device)
    ENQUEUE_OFFSET(reset_field_device)
}

