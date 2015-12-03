#include "ctx_common.hpp"

extern "C" void field_summary_kernel_ocl_
(double* vol, double* mass, double* ie, double* temp)
{
    tea_context.field_summary_kernel(vol, mass, ie, temp);
}

void TeaCLContext::field_summary_kernel
(double* vol, double* mass, double* ie, double* temp)
{
    chunks.at(fine_chunk)->field_summary_kernel(vol, mass, ie, temp);
}

