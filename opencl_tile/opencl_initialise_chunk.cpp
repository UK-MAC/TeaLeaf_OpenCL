#include "../ctx_common.hpp"

void TeaOpenCLChunk::initialise_chunk_kernel
(double d_xmin, double d_ymin, double d_dx, double d_dy)
{
    launch_specs.at("initialise_chunk_first_device").offset = cl::NullRange;

    initialise_chunk_first_device.setArg(1, d_xmin);
    initialise_chunk_first_device.setArg(2, d_ymin);
    initialise_chunk_first_device.setArg(3, d_dx);
    initialise_chunk_first_device.setArg(4, d_dy);

    initialise_chunk_second_device.setArg(1, d_dx);
    initialise_chunk_second_device.setArg(2, d_dy);

    ENQUEUE(initialise_chunk_first_device);
    ENQUEUE(initialise_chunk_second_device);
}

