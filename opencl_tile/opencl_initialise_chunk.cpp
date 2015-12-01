#include "../ocl_common.hpp"

void TeaOpenCLTile::initialise_chunk_kernel
(double d_xmin, double d_ymin, double d_dx, double d_dy)
{
    launch_specs.at("initialise_chunk_first_device").offset = cl::NullRange;
    initialise_chunk_first_device.setArg(8, d_xmin);
    initialise_chunk_first_device.setArg(9, d_ymin);
    initialise_chunk_first_device.setArg(10, d_dx);
    initialise_chunk_first_device.setArg(11, d_dy);

    initialise_chunk_second_device.setArg(0, d_dx);
    initialise_chunk_second_device.setArg(1, d_dy);

    ENQUEUE(initialise_chunk_first_device);
    ENQUEUE(initialise_chunk_second_device);
}

