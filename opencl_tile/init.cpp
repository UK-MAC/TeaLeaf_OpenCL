#include "../ocl_common.hpp"

TeaOpenCLTile::TeaTile
(run_params_t run_params, cl::Context context, cl::Device device)
:device(device),
 context(context),
 run_params(run_params)
{
    ;
    // FIXME need to initialise the x_cells, y_cells, maybe some others

    MPI_Comm_rank(MPI_COMM_WORLD, &rank);

    int device_type = device.getInfo<CL_DEVICE_TYPE>();

    // choose reduction based on device type
    switch (device_type)
    {
    case CL_DEVICE_TYPE_GPU : 
        device_type_prepro = "-DCL_DEVICE_TYPE_GPU ";
        break;
    case CL_DEVICE_TYPE_CPU : 
        device_type_prepro = "-DCL_DEVICE_TYPE_CPU ";
        break;
    case CL_DEVICE_TYPE_ACCELERATOR : 
        device_type_prepro = "-DCL_DEVICE_TYPE_ACCELERATOR ";
        break;
    default :
        device_type_prepro = "-DCL_DEVICE_TYPE_GPU ";
        break;
    }

    std::string devname;
    device.getInfo(CL_DEVICE_NAME, &devname);

    //fprintf(stdout, "OpenCL using device %d (%s) in rank %d\n",
    //    actual_device, devname.c_str(), rank);

    // initialise command queue
    if (run_params.profiler_on)
    {
        // turn on profiling
        queue = cl::CommandQueue(context, device,
                                 CL_QUEUE_PROFILING_ENABLE, NULL);
    }
    else
    {
        queue = cl::CommandQueue(context, device);
    }

    initProgram();
    initSizes();
    initReduction();
    initMemory();
    initArgs();
}

