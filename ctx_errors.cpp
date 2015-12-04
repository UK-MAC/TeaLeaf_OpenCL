#include "mpi.h"
#include "ctx_common.hpp"

#include <cstdio>
#include <cstdarg>
#include <numeric>

// called when something goes wrong
void cloverDie
(int line, const char* filename, const char* format, ...)
{
    fprintf(stderr, "@@@@@\n");
    fprintf(stderr, "\x1b[31m");
    fprintf(stderr, "Fatal error at line %d in %s:", line, filename);
    fprintf(stderr, "\x1b[0m");
    fprintf(stderr, "\n");

    va_list arglist;
    va_start(arglist, format);
    vfprintf(stderr, format, arglist);
    va_end(arglist);

    // TODO add logging or something

    fprintf(stderr, "\nExiting\n");

    MPI_Abort(MPI_COMM_WORLD, 1);
}

extern "C" void print_opencl_profiling_info_
(void)
{
    tea_context.print_profiling_info();
}

// print out timing info when done
void TeaCLContext::print_profiling_info
(void)
{
    if (run_params.profiler_on)
    {
        fprintf(stdout, "@@@@@ OpenCL Profiling information (from rank 0) @@@@@\n");

        std::map<std::string, double> all_kernel_times;
        std::map<std::string, int> all_kernel_calls;

        std::map<std::string, double>::iterator ii;
        std::map<std::string, int>::iterator jj;

        FOR_EACH_CHUNK
        {
            for (ii = chunk_it->second->kernel_times.begin(), jj = chunk_it->second->kernel_calls.begin();
                ii != chunk_it->second->kernel_times.end(); ii++, jj++)
            {
                std::string func_name = ii->first;

                if (all_kernel_times.end() != all_kernel_times.find(func_name))
                {
                    all_kernel_calls.at(func_name) += jj->second;
                    all_kernel_times.at(func_name) += ii->second;
                }
                else
                {
                    all_kernel_calls[func_name] = jj->second;
                    all_kernel_times[func_name] = ii->second;
                }
            }
        }

        double total_time;
        int total_calls;

        MPI_Reduce(&ii->second, &total_time, 1, MPI_DOUBLE, MPI_SUM, 0, MPI_COMM_WORLD);
        MPI_Reduce(&jj->second, &total_calls, 1, MPI_INT, MPI_SUM, 0, MPI_COMM_WORLD);

        if (!rank && total_calls)
        {
            fprintf(stdout, "%30s : %10.3f ms (%.2f μs avg. over %d calls)\n",
                ii->first.c_str(), total_time, 1e3*total_time/total_calls, total_calls);
        }
    }
}

