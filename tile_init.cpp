#include "ocl_common.hpp"

void TeaCLTile::initTileQueue
(int device_id, bool profiler_on, cl::Context context)
{

    if (actual_device >= devices.size())
    {
        DIE("Device %d was selected in rank %d but there are only %zu available\n",
            actual_device, rank, devices.size());
    }
    else
    {
        device = devices.at(actual_device);
    }

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

