#include "ocl_common.hpp"
extern CloverChunk chunk;

extern "C" void set_field_kernel_ocl_
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
    chunk.set_field_kernel();
}

void CloverChunk::set_field_kernel
(void)
{
    ENQUEUE(set_field_device)
}

