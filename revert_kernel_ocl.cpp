#include "ocl_common.hpp"
extern CloverChunk chunk;

extern "C" void revert_kernel_ocl_
(int *x_min,int *x_max,int *y_min,int *y_max,
const double* density0,
      double* density1,
const double* energy0,
      double* energy1)
{
    chunk.revert_kernel();
}

void CloverChunk::revert_kernel
(void)
{
    //ENQUEUE(revert_device)
    ENQUEUE_OFFSET(revert_device)
}

