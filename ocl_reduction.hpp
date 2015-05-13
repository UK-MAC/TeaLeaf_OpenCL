#ifndef __CL_REDUCTION_HDR
#define __CL_REDUCTION_HDR

#include "ocl_common.hpp"

#include <numeric>

template <typename T>
void TeaCLTile::reduceValue
(reduce_info_vec_t& red_kernels,
 const cl::Buffer& results_buf,
 T* result, cl::Event* copy_event, cl::Event* kernel_event)
{
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
            kernel_event);
    }

    // copy back the result and return
#if 0
    queue.enqueueReadBuffer(results_buf,
                            CL_TRUE,
                            0,
                            sizeof(T),
                            result,
                            NULL,
                            copy_event);
#else
    int err;
    err = clEnqueueReadBuffer(queue(),
        results_buf(),
        CL_FALSE,
        0,
        sizeof(T),
        result,
        1,
        &(*kernel_event)(),
        &(*copy_event)());

    if (CL_SUCCESS != err)
    {
        DIE("Error number %d in copying back reduction result", err);
    }
#endif
}

template <typename T>
void TeaCLTile::sumReduceValue
(int buffer, T* result, cl::Event* copy_event, cl::Event* kernel_event)
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

    reduceValue(sum_red_kernels_double, results_buf, result, copy_event, kernel_event);
}

template <typename T>
std::vector<T> TeaCLContext::sumReduceValues
(const std::vector<int>& buffer_indexes)
{
    // one vector per buffer to be reduced, each with one vector per tile
    std::vector<std::vector<T> > reduced_values(buffer_indexes.size(), std::vector<T>(tiles.size()));
    std::vector<std::vector<cl::Event> > copy_events(buffer_indexes.size(), std::vector<cl::Event>(tiles.size()));
    std::vector<std::vector<cl::Event> > kernel_events(buffer_indexes.size(), std::vector<cl::Event>(tiles.size()));

    FOR_EACH_TILE
    {
        tile->queue.finish();
    }

    for (size_t ii = 0; ii < buffer_indexes.size(); ii++)
    {
        for (size_t tt = 0; tt < tiles.size(); tt++)
        {
            /*
             *  for each buffer that needs to be reduced, make each tile
             *  individually do the reduction and enqueue a copy into the
             *  relative part of the array.
             */
            TeaCLTile tile = tiles.at(tt);
            tile.sumReduceValue<T>(
                buffer_indexes.at(ii),
                &reduced_values.at(ii).at(tt),
                &copy_events.at(ii).at(tt),
                &kernel_events.at(ii).at(tt));
        }
    }

    std::vector<T> results(buffer_indexes.size());

    for (size_t ii = 0; ii < buffer_indexes.size(); ii++)
    {
        /*
         *  Then, for each element that needs to be reduced, wait on events for
         *  all tiles and do a quick local reduction
         */
        cl::Event::waitForEvents(copy_events.at(ii));
        results.at(ii) = std::accumulate(reduced_values.at(ii).begin(), reduced_values.at(ii).end(), T(0.0));
    }

    return results;
}

#endif

