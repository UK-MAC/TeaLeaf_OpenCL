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
    //ENQUEUE(field_summary_device);
    ENQUEUE_OFFSET(field_summary_device);

    std::vector<int> indexes;
    indexes.push_back(1);
    indexes.push_back(2);
    indexes.push_back(3);
    indexes.push_back(4);
    std::vector<double> reduced_values = sumReduceValues<double>(indexes);

    *vol = reduced_values.at(0);
    *mass = reduced_values.at(1);
    *ie = reduced_values.at(2);
    *temp = reduced_values.at(3);
}

