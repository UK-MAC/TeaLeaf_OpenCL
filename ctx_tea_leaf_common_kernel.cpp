#include "ocl_common.hpp"
#include "ocl_reduction.hpp"

#include <cmath>

extern "C" void tea_leaf_calc_2norm_kernel_ocl_
(int* norm_array, double* norm)
{
    tea_context.tea_leaf_calc_2norm_kernel(*norm_array, norm);
}

extern "C" void tea_leaf_common_init_kernel_ocl_
(const int * coefficient, double * dt, double * rx, double * ry,
 int * zero_boundary, int * reflective_boundary)
{
    tea_context.tea_leaf_common_init(*coefficient, *dt, rx, ry,
        zero_boundary, *reflective_boundary);
}

extern "C" void tea_leaf_common_finalise_kernel_ocl_
(void)
{
    tea_context.tea_leaf_finalise();
}

extern "C" void tea_leaf_calc_residual_ocl_
(void)
{
    tea_context.tea_leaf_calc_residual();
}

/********************/

// copy back dx/dy and calculate rx/ry
void TeaCLContext::calcrxry
(double dt, double * rx, double * ry)
{
    tiles.at(fine_tile)->calcrxry(dt, rx, ry);
}

void TeaCLContext::tea_leaf_calc_2norm_kernel
(int norm_array, double* norm)
{
    tiles.at(fine_tile)->tea_leaf_calc_2norm_kernel(norm_array, norm);
}

void TeaCLContext::tea_leaf_common_init
(int coefficient, double dt, double * rx, double * ry,
 int * zero_boundary, int reflective_boundary)
{
    if (coefficient != COEF_CONDUCTIVITY && coefficient != COEF_RECIP_CONDUCTIVITY)
    {
        DIE("Unknown coefficient %d passed to tea leaf\n", coefficient);
    }

    calcrxry(dt, rx, ry);

    tiles.at(fine_tile)->tea_leaf_common_init(coefficient, dt, rx, ry,
        zero_boundary, reflective_boundary);
}

void TeaCLContext::tea_leaf_finalise
(void)
{
    tiles.at(fine_tile)->tea_leaf_finalise();
}

void TeaCLContext::tea_leaf_calc_residual
(void)
{
    tiles.at(fine_tile)->tea_leaf_calc_residual();
}

/********************/

