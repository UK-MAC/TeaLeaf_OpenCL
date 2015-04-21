#include "ocl_common.hpp"

extern "C" void initialise_chunk_kernel_ocl_
(double* d_xmin, double* d_ymin, double* d_dx, double* d_dy)
{
    chunk.initialise_chunk_kernel(*d_xmin, *d_ymin, *d_dx, *d_dy);
}

void CloverChunk::initialise_chunk_kernel
(double d_xmin, double d_ymin, double d_dx, double d_dy)
{
    launch_specs.at("initialise_chunk_first_device").offset = cl::NullRange;
    initialise_chunk_first_device.setArg(0, d_xmin);
    initialise_chunk_first_device.setArg(1, d_ymin);
    initialise_chunk_first_device.setArg(2, d_dx);
    initialise_chunk_first_device.setArg(3, d_dy);
    ENQUEUE_OFFSET(initialise_chunk_first_device)

    initialise_chunk_second_device.setArg(0, d_xmin);
    initialise_chunk_second_device.setArg(1, d_ymin);
    initialise_chunk_second_device.setArg(2, d_dx);
    initialise_chunk_second_device.setArg(3, d_dy);
    ENQUEUE_OFFSET(initialise_chunk_second_device)
    queue.finish();
}

