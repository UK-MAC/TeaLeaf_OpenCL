#include "mpi.h"
#include "ctx_common.hpp"
#include "ctx_strings.hpp"

#include <sstream>
#include <fstream>
#include <iostream>
#include <algorithm>

TeaCLContext tea_context;

extern "C" void initialise_ocl_
(void)
{
    tea_context = TeaCLContext();
    tea_context.initialise();
}

extern "C" void timer_c_(double*);

void TeaCLContext::initialise
(void)
{
    MPI_Comm_rank(MPI_COMM_WORLD, &rank);

    double t0;
    timer_c_(&t0);

    if (!rank)
    {
        fprintf(stdout, "Initialising OpenCL\n");
    }

    initOcl();

    MPI_Barrier(MPI_COMM_WORLD);

    if (!rank)
    {
        double t1;
        timer_c_(&t1);

        fprintf(stdout, "Finished initialisation in %f seconds\n", t1-t0);
    }
}

static void stripString
(std::string & input_string)
{
    // trim whitespace from a string
    input_string.erase(input_string.find_last_not_of(" \n\r\t")+1);
    input_string.erase(input_string.begin(), input_string.begin()+input_string.find_first_not_of(" \n\r\t"));
}

static void listPlatforms
(std::vector<cl::Platform>& platforms)
{
    for (size_t pp = 0; pp < platforms.size(); pp++)
    {
        std::string profile, version, name, vendor;
        platforms.at(pp).getInfo(CL_PLATFORM_PROFILE, &profile);
        platforms.at(pp).getInfo(CL_PLATFORM_VERSION, &version);
        platforms.at(pp).getInfo(CL_PLATFORM_NAME, &name);
        platforms.at(pp).getInfo(CL_PLATFORM_VENDOR, &vendor);

        fprintf(stdout, "Platform %zu: %s - %s (profile = %s, version = %s)\n",
            pp, vendor.c_str(), name.c_str(), profile.c_str(), version.c_str());

        std::vector<cl::Device> devices;
        platforms.at(pp).getDevices(CL_DEVICE_TYPE_ALL, &devices);

        for (size_t ii = 0; ii < devices.size(); ii++)
        {
            std::string devname;
            cl_device_type dtype;
            devices.at(ii).getInfo(CL_DEVICE_NAME, &devname);
            devices.at(ii).getInfo(CL_DEVICE_TYPE, &dtype);
            stripString(devname);

            std::string dtype_str = strType(dtype);
            fprintf(stdout, " Device %zu: %s (%s)\n", ii, devname.c_str(), dtype_str.c_str());
        }
    }
}

void TeaCLContext::initOcl
(void)
{
    std::vector<cl::Platform> platforms;

    try
    {
        cl::Platform::get(&platforms);
    }
    catch (cl::Error e)
    {
        DIE("Error in fetching platforms (%s), error %d\n", e.what(), e.err());
    }

    if (platforms.size() < 1)
    {
        DIE("No platforms found\n");
    }

    // Read in from file - easier than passing in from fortran
    std::ifstream input("tea.in");
    input.exceptions(std::ifstream::badbit);

    if (!input.is_open())
    {
        // should never happen
        DIE("Input file not found\n");
    }

    // use first device whatever happens (ignore MPI rank) for running across different platforms
    bool usefirst = paramEnabled(input, "opencl_usefirst");

    run_params.profiler_on = paramEnabled(input, "profiler_on");

    std::string desired_vendor = readString(input, "opencl_vendor");

    int preferred_device = readInt(input, "opencl_device");
    preferred_device = (preferred_device < 0) ? 0 : preferred_device;
    fprintf(DBGOUT, "Preferred device is %d\n", preferred_device);

    std::string type_name = readString(input, "opencl_type");
    int desired_type = typeMatch(type_name);

    int file_halo_depth = readInt(input, "halo_depth");

    // FIXME levels
    int file_n_tiles = readInt(input, "tiles");

    // No error checking - assume fortran does it correctly
    run_params.halo_exchange_depth = file_halo_depth;

    bool tl_use_jacobi = paramEnabled(input, "tl_use_jacobi");
    bool tl_use_cg = paramEnabled(input, "tl_use_cg");
    bool tl_use_chebyshev = paramEnabled(input, "tl_use_chebyshev");
    bool tl_use_ppcg = paramEnabled(input, "tl_use_ppcg");
    bool tl_use_dpcg = paramEnabled(input, "tl_use_dpcg");

    // set solve
    if(!rank)fprintf(stdout, "Solver to use: ");
    if (tl_use_dpcg)
    {
        run_params.tea_solver = TEA_ENUM_DPCG;
        if(!rank)fprintf(stdout, "DPCG\n");
    }
    else if (tl_use_ppcg)
    {
        run_params.tea_solver = TEA_ENUM_PPCG;
        if(!rank)fprintf(stdout, "PPCG\n");
    }
    else if (tl_use_chebyshev)
    {
        run_params.tea_solver = TEA_ENUM_CHEBYSHEV;
        if(!rank)fprintf(stdout, "Chebyshev + CG\n");
    }
    else if (tl_use_cg)
    {
        run_params.tea_solver = TEA_ENUM_CG;
        if(!rank)fprintf(stdout, "Conjugate gradient\n");
    }
    else if (tl_use_jacobi)
    {
        run_params.tea_solver = TEA_ENUM_JACOBI;
        if(!rank)fprintf(stdout, "Jacobi\n");
    }
    else
    {
        run_params.tea_solver = TEA_ENUM_JACOBI;
        if(!rank)fprintf(stdout, "Jacobi (no solver specified in tea.in)\n");
    }

    std::string desired_preconditioner = readString(input, "tl_preconditioner_type");

    // set preconditioner type
    if(!rank)fprintf(stdout, "Preconditioner to use: ");
    if (desired_preconditioner.find("jac_diag") != std::string::npos)
    {
        run_params.preconditioner_type = TL_PREC_JAC_DIAG;
        if(!rank)fprintf(stdout, "Diagonal Jacobi\n");
    }
    else if (desired_preconditioner.find("jac_block") != std::string::npos)
    {
        run_params.preconditioner_type = TL_PREC_JAC_BLOCK;
        if(!rank)fprintf(stdout, "Block Jacobi\n");
    }
    else if (desired_preconditioner.find("none") != std::string::npos)
    {
        run_params.preconditioner_type = TL_PREC_NONE;
        if(!rank)fprintf(stdout, "None\n");
    }
    else
    {
        run_params.preconditioner_type = TL_PREC_NONE;
        if(!rank)fprintf(stdout, "None (no preconditioner specified in tea.in)\n");
    }

    if (desired_vendor.find("no_setting") != std::string::npos)
    {
        DIE("No opencl_vendor specified in tea.in\n");
    }
    else if (desired_vendor.find("list") != std::string::npos)
    {
        // special case to print out platforms instead
        fprintf(stdout, "Listing platforms\n\n");
        listPlatforms(platforms);
        exit(0);
    }
    else if (desired_vendor.find("any") != std::string::npos)
    {
        fprintf(stdout, "Choosing first platform that matches device type\n");

        cl::Platform platform;

        // go through all platforms
        for (size_t ii = 0; ; ii++)
        {
            // if there are no platforms left to match
            if (platforms.size() == ii)
            {
                fprintf(stderr, "Platforms available:\n");

                listPlatforms(platforms);

                DIE("No platform with specified device type was found\n");
            }

            std::vector<cl::Device> devices;

            try
            {
                platforms.at(ii).getDevices(desired_type, &devices);
            }
            catch (cl::Error e)
            {
                if (e.err() == CL_DEVICE_NOT_FOUND)
                {
                    continue;
                }
                else
                {
                    DIE("Error %d (%s) in querying devices\n", e.err(), e.what());
                }
            }

            if (devices.size() > 0)
            {
                platform = platforms.at(ii);

                std::vector<cl::Platform> used(1, platform);
                fprintf(stdout, "Using platform:\n");
                listPlatforms(used);

                // try to create a context with the desired type
                cl_context_properties properties[3] = {CL_CONTEXT_PLATFORM,
                    reinterpret_cast<cl_context_properties>(platform()), 0};

                try
                {
                    context = cl::Context(desired_type, properties);
                }
                catch (cl::Error e)
                {
                    DIE("Error %d (%s) in creating context\n", e.err(), e.what());
                }

                break;
            }
        }
    }
    else
    {
        cl::Platform platform;

        // go through all platforms
        for (size_t ii = 0; ; )
        {
            std::string plat_name;
            platforms.at(ii).getInfo(CL_PLATFORM_VENDOR, &plat_name);
            std::transform(plat_name.begin(),
                           plat_name.end(),
                           plat_name.begin(),
                           tolower);
            fprintf(DBGOUT, "Checking platform %s\n", plat_name.c_str());

            // if the platform name given matches one in the LUT
            if (plat_name.find(desired_vendor) != std::string::npos)
            {
                fprintf(DBGOUT, "Correct vendor platform found\n");
                platform = platforms.at(ii);

                std::vector<cl::Platform> used(1, platform);
                fprintf(stdout, "Using platform:\n");
                listPlatforms(used);
                break;
            }
            else if (platforms.size() == ++ii)
            {
                // if there are no platforms left to match
                fprintf(stderr, "Platforms available:\n");

                listPlatforms(platforms);

                DIE("Correct vendor platform NOT found\n");
            }
        }

        // try to create a context with the desired type
        cl_context_properties properties[3] = {CL_CONTEXT_PLATFORM,
            reinterpret_cast<cl_context_properties>(platform()), 0};

        try
        {
            cl::Context test_context(desired_type, properties);
        }
        catch (cl::Error e)
        {
            if (e.err() == CL_DEVICE_NOT_AVAILABLE)
            {
                DIE("Devices found but are not available (CL_DEVICE_NOT_AVAILABLE)\n");
            }
            // if there's no device of the desired type in this context
            else if (e.err() == CL_DEVICE_NOT_FOUND)
            {
                fprintf(stderr, "No devices of specified type (%s) found in platform.\n", strType(desired_type).c_str());
                fprintf(stderr, "Platforms available:\n");
                listPlatforms(platforms);

                DIE("Unable to get devices of desired type on platform");
            }
            else
            {
                DIE("Error %d (%s) in creating context\n", e.err(), e.what());
            }
        }
    }

    // gets devices one at a time to prevent conflicts (on emerald)
    int ranks, cur_rank = 0;

    MPI_Comm_size(MPI_COMM_WORLD, &ranks);
    MPI_Comm_rank(MPI_COMM_WORLD, &rank);

    do
    {
        if (rank == cur_rank)
        {
            // index of device to use
            int actual_device = 0;

            // get devices - just choose the first one
            std::vector<cl::Device> devices;
            context.getInfo(CL_CONTEXT_DEVICES, &devices);

            cl::Device device;

            if (usefirst)
            {
                // always use specified device and ignore rank
                actual_device = preferred_device;
            }
            else
            {
                actual_device = preferred_device + (rank % devices.size());
            }

            if (preferred_device < 0)
            {
                // if none specified or invalid choice, choose 0
                fprintf(stdout, "No device specified, choosing device 0\n");
                actual_device = 0;
                device = devices.at(actual_device);
            }
            else if (actual_device >= static_cast<int>(devices.size()))
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

            // TODO should be able to use a fortran one instead
            // TODO need to pass size of chunk (calculated in Fortran)
            tile_ptr_t new_tile(new TeaOpenCLTile(run_params, context, device));

            tiles[fine_tile] = new_tile;
            // FIXME create coarse tile
        }
        MPI_Barrier(MPI_COMM_WORLD);
    } while ((cur_rank++) < ranks);

    if (!rank)
    {
        fprintf(stdout, "Finished creating tiles\n");
    }

    MPI_Barrier(MPI_COMM_WORLD);
}

