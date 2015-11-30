#include "ocl_common.hpp"

extern "C" void set_field_kernel_ocl_
(void)
{
    tea_context.set_field_kernel();
}

void TeaCLContext::set_field_kernel
(void)
{
    FOR_EACH_TILE
    {
        ENQUEUE(set_field_device)
    }
}

