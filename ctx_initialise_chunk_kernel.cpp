#include "ctx_common.hpp"

extern "C" void initialise_chunk_kernel_ocl_
(double* d_xmin, double* d_ymin, double* d_dx, double* d_dy)
{
    tea_context.initialise_chunk_kernel(*d_xmin, *d_ymin, *d_dx, *d_dy);
}

void TeaCLContext::initialise_chunk_kernel
(double d_xmin, double d_ymin, double d_dx, double d_dy)
{
    chunks.at(fine_chunk)->initialise_chunk_kernel(d_xmin, d_ymin, d_dx, d_dy);
}

