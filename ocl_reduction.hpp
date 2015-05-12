#ifndef __CL_REDUCTION_HDR
#define __CL_REDUCTION_HDR

#include "ocl_common.hpp"

#include <numeric>

template <typename T>
void TeaCLTile::reduceValue
(reduce_info_vec_t& red_kernels,
 const cl::Buffer& results_buf,
 T* result, cl::Event* copy_event)
{
    std::vector<cl::Event> kernel_events(red_kernels.size());

    // enqueue the kernels in order
    for (size_t ii = 0; ii < red_kernels.size(); ii++)
    {
        red_kernels.at(ii).kernel.setArg(0, results_buf);

        enqueueKernel(red_kernels.at(ii).kernel,
            __LINE__, __FILE__,
            cl::NullRange,
            red_kernels.at(ii).global_size,
            red_kernels.at(ii).local_size,
            NULL,
            &kernel_events.at(ii));
    }

    // copy back the result and return
    queue.enqueueReadBuffer(results_buf,
                            CL_TRUE,
                            0,
                            sizeof(T),
                            &result,
                            &kernel_events,
                            copy_event);
}

template <typename T>
void TeaCLTile::sumReduceValue
(int buffer, T* result, cl::Event* copy_event)
{
    cl::Buffer results_buf;

    switch (buffer)
    {
    case 1:
        results_buf = reduce_buf_1;
        break;
    case 2:
        results_buf = reduce_buf_2;
        break;
    case 3:
        results_buf = reduce_buf_3;
        break;
    case 4:
        results_buf = reduce_buf_4;
        break;
    case 5:
        results_buf = reduce_buf_5;
        break;
    case 6:
        results_buf = reduce_buf_6;
        break;
    default:
        DIE("Invalid buffer index %d passed to reduceValue", buffer);
    }

    reduceValue(sum_red_kernels_double, results_buf, result, copy_event);
}

template <typename T>
std::vector<T> TeaCLContext::sumReduceValues
(const std::vector<int>& buffer_indexes)
{
    // one vector per buffer to be reduced, each with one vector per tile
    std::vector<std::vector<T> > reduced_values(buffer_indexes.size(), std::vector<T>(tiles.size()));
    std::vector<std::vector<cl::Event> > copy_events(buffer_indexes.size(), std::vector<cl::Event>(tiles.size()));

    for (size_t ii = 0; ii < buffer_indexes.size(); ii++)
    {
        for (size_t tt = 0; tt < tiles.size(); tt++)
        {
            TeaCLTile tile = tiles.at(tt);
            tile.sumReduceValue<T>(
                buffer_indexes.at(ii),
                &reduced_values.at(ii).at(tt),
                &copy_events.at(ii).at(tt));
        }
    }

    std::vector<T> results(buffer_indexes.size());

    for (size_t ii = 0; ii < buffer_indexes.size(); ii++)
    {
        cl::Event::waitForEvents(copy_events.at(ii));
        results.at(ii) = std::accumulate(reduced_values.at(ii).begin(), reduced_values.at(ii).end(), 0.0);
    }

    return results;
}

#endif

