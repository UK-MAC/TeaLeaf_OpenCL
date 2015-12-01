#include "ocl_common.hpp"

void TeaOpenCLTile::field_summary_kernel
(double* vol, double* mass, double* ie, double* temp)
{
    ENQUEUE(field_summary_device);

    *vol = reduceValue<double>(sum_red_kernels_double, reduce_buf_1);
    *mass = reduceValue<double>(sum_red_kernels_double, reduce_buf_2);
    *ie = reduceValue<double>(sum_red_kernels_double, reduce_buf_3);
    *temp = reduceValue<double>(sum_red_kernels_double, reduce_buf_4);
}

