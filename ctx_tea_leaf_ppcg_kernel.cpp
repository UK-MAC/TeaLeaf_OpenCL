#include "ctx_common.hpp"

extern "C" void tea_leaf_ppcg_init_ocl_
(const double * ch_alphas, const double * ch_betas,
 double* theta, int* n_inner_steps)
{
    tea_context.ppcg_init(ch_alphas, ch_betas, *theta, *n_inner_steps);
}

extern "C" void tea_leaf_ppcg_init_sd_kernel_ocl_
(void)
{
    tea_context.ppcg_init_sd_kernel();
}

extern "C" void tea_leaf_ppcg_inner_kernel_ocl_
(int * inner_step,
 int * bounds_extra,
 const int* chunk_neighbours)
{
    tea_context.tea_leaf_ppcg_inner_kernel(*inner_step, *bounds_extra, chunk_neighbours);
}

/********************/

void TeaCLContext::ppcg_init
(const double * ch_alphas, const double * ch_betas,
 const double theta, const int n_inner_steps)
{
    tiles.at(fine_tile)->ppcg_init(ch_alphas, ch_betas, theta, n_inner_steps);
}

void TeaCLContext::ppcg_init_sd_kernel
(void)
{
    tiles.at(fine_tile)->ppcg_init_sd_kernel();
}

void TeaCLContext::tea_leaf_ppcg_inner_kernel
(int inner_step, int bounds_extra, const int* chunk_neighbours)
{
    tiles.at(fine_tile)->tea_leaf_ppcg_inner_kernel(inner_step, bounds_extra,
        chunk_neighbours);
}

