#include "ocl_common.hpp"
#include "ocl_reduction.hpp"

extern "C" void tea_leaf_cheby_init_kernel_ocl_
(const double * ch_alphas, const double * ch_betas, int* n_coefs,
 const double * rx, const double * ry, const double * theta)
{
    tea_context.tea_leaf_cheby_init_kernel(ch_alphas, ch_betas, *n_coefs,
        *rx, *ry, *theta);
}

extern "C" void tea_leaf_cheby_iterate_kernel_ocl_
(const int * cheby_calc_step)
{
    tea_context.tea_leaf_cheby_iterate_kernel(*cheby_calc_step);
}

/********************/

void TeaCLContext::tea_leaf_cheby_init_kernel
(const double * ch_alphas, const double * ch_betas, int n_coefs,
 const double rx, const double ry, const double theta)
{
    tiles.at(fine_tile)->tea_leaf_cheby_init_kernel(ch_alphas, ch_betas,
        n_coefs, rx, ry, theta);
}

void TeaCLContext::tea_leaf_cheby_iterate_kernel
(const int cheby_calc_step)
{
    tiles.at(fine_tile)->tea_leaf_cheby_iterate_kernel(cheby_calc_step);
}

