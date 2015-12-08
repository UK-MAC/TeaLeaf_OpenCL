#include "../ctx_common.hpp"

TeaOpenCLChunk::TeaOpenCLChunk
(run_params_t run_params, cl::Context context, cl::Device device,
 int x_cells, int y_cells, int coarse_x_cells, int coarse_y_cells)
:device(device),
 context(context),
 run_params(run_params),
 chunk_x_cells(x_cells), chunk_y_cells(y_cells),
 local_coarse_x_cells(coarse_x_cells),local_coarse_y_cells(coarse_y_cells)
{
    //fprintf(stdout, "%d %d\n", chunk_x_cells, chunk_y_cells);

    MPI_Comm_rank(MPI_COMM_WORLD, &rank);

    int device_type = device.getInfo<CL_DEVICE_TYPE>();

    // choose reduction based on device type
    switch (device_type)
    {
    case CL_DEVICE_TYPE_GPU : 
        device_type_prepro = "-D CL_DEVICE_TYPE_GPU ";
        break;
    case CL_DEVICE_TYPE_CPU : 
        device_type_prepro = "-D CL_DEVICE_TYPE_CPU ";
        break;
    case CL_DEVICE_TYPE_ACCELERATOR : 
        device_type_prepro = "-D CL_DEVICE_TYPE_ACCELERATOR ";
        break;
    default :
        device_type_prepro = "-D CL_DEVICE_TYPE_GPU ";
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

