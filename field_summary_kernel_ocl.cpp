#include "ocl_common.hpp"
#include "ocl_reduction.hpp"

extern "C" void field_summary_kernel_ocl_
(double* vol, double* mass, double* ie, double* temp)
{
    tea_context.field_summary_kernel(vol, mass, ie, temp);
}

void TeaCLContext::field_summary_kernel
(double* vol, double* mass, double* ie, double* temp)
{
    FOR_EACH_TILE
    {
        ENQUEUE(field_summary_device);
    }

    int vindexes[] = {1, 2, 3, 4};
    std::vector<int> indexes(vindexes, vindexes+4);
    std::vector<double> reduced_values = sumReduceValues<double>(indexes);

    *vol = reduced_values.at(0);
    *mass = reduced_values.at(1);
    *ie = reduced_values.at(2);
    *temp = reduced_values.at(3);
}

