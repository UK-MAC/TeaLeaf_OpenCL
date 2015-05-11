#include "ocl_common.hpp"

TeaCLTile::TeaCLTile
(int* in_x_min, int* in_x_max,
 int* in_y_min, int* in_y_max)
:x_min(*in_x_min),
 x_max(*in_x_max),
 y_min(*in_y_min),
 y_max(*in_y_max)

void TeaCLTile::initTileQueue
(bool profiler_on, cl::Device chosen_device, cl::Context context)
{
    device = chosen_device;

    std::string devname;
    device.getInfo(CL_DEVICE_NAME, &devname);

    fprintf(stdout, "OpenCL using device %d (%s) in rank %d\n",
        actual_device, devname.c_str(), rank);

    // initialise command queue
    if (profiler_on)
    {
        // turn on profiling
        queue = cl::CommandQueue(context, device,
                                 CL_QUEUE_PROFILING_ENABLE, NULL);
    }
    else
    {
        queue = cl::CommandQueue(context, device);
    }
}

