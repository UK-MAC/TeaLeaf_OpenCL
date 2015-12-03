#include "ctx_common.hpp"

extern "C" void set_field_kernel_ocl_
(void)
{
    tea_context.set_field_kernel();
}

void TeaCLContext::set_field_kernel
(void)
{
    chunks.at(fine_chunk)->set_field_kernel();
}

