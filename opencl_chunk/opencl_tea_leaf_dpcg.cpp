#include "../ctx_common.hpp"
#include "opencl_reduction.hpp"

#include <cmath>

// FIXME some of these might not have to copy memory back and forth as much as they do

void TeaOpenCLChunk::tea_leaf_dpcg_coarsen_matrix_kernel
(double * host_Kx, double * host_Ky)
{
    ENQUEUE_DEFLATION(tea_leaf_dpcg_coarsen_matrix_device);

    queue.finish();

    queue.enqueueReadBuffer(coarse_local_Kx, CL_TRUE, 0,
        local_coarse_x_cells*local_coarse_y_cells*sizeof(double),
        host_Kx);

    queue.enqueueReadBuffer(coarse_local_Ky, CL_TRUE, 0,
        local_coarse_x_cells*local_coarse_y_cells*sizeof(double),
        host_Ky);
}

void TeaOpenCLChunk::getCoarseCopyParameters
(cl::size_t<3> * buffer_origin,
 cl::size_t<3> * host_origin,
 cl::size_t<3> * region,
 size_t * buffer_row_pitch,
 size_t * host_row_pitch)
{
    // copying from the host, needs to take halos into account
    (*host_origin)[0] = run_params.halo_exchange_depth;
    (*host_origin)[1] = run_params.halo_exchange_depth;
    (*host_origin)[2] = 0;

    (*buffer_origin)[0] = run_params.halo_exchange_depth;
    (*buffer_origin)[1] = run_params.halo_exchange_depth;
    (*buffer_origin)[2] = 0;

    (*region)[0] = chunk_x_cells;
    (*region)[1] = chunk_y_cells;
    (*region)[2] = 1;

    // convert to bytes
    (*host_origin)[0] *= sizeof(double);
    (*buffer_origin)[0] *= sizeof(double);
    (*region)[0] *= sizeof(double);

    (*buffer_row_pitch) = (chunk_x_cells + 2*run_params.halo_exchange_depth)*sizeof(double);
    (*host_row_pitch) = (chunk_x_cells + 2*run_params.halo_exchange_depth)*sizeof(double);

}

void TeaOpenCLChunk::tea_leaf_dpcg_copy_reduced_coarse_grid
(double * global_coarse_Kx, double * global_coarse_Ky, double * global_coarse_Di)
{
    cl::size_t<3> buffer_origin;
    cl::size_t<3> host_origin;
    cl::size_t<3> region;

    size_t buffer_row_pitch;
    size_t host_row_pitch;

    getCoarseCopyParameters(&buffer_origin, &host_origin, &region,
        &buffer_row_pitch, &host_row_pitch);

    // Need to copy back into the middle of the grid, not in the halos
    queue.enqueueWriteBufferRect(vector_Kx, CL_TRUE,
        buffer_origin,
        host_origin,
        region,
        buffer_row_pitch,
        0,
        host_row_pitch,
        0,
        global_coarse_Kx);
    queue.enqueueWriteBufferRect(vector_Ky, CL_TRUE,
        buffer_origin,
        host_origin,
        region,
        buffer_row_pitch,
        0,
        host_row_pitch,
        0,
        global_coarse_Ky);
    // FIXME diagonal...?
    //queue.enqueueWriteBufferRect(vector_Di, CL_TRUE,
    //    buffer_origin,
    //    host_origin,
    //    region,
    //    buffer_row_pitch,
    //    0,
    //    host_row_pitch,
    //    0,
    //    global_coarse_Di);
}

void TeaOpenCLChunk::tea_leaf_dpcg_prolong_z_kernel
(double * t2_local)
{
    queue.enqueueWriteBuffer(coarse_local_t2, CL_TRUE, 0, 
        local_coarse_x_cells*local_coarse_y_cells*sizeof(double),
        t2_local);

    ENQUEUE_DEFLATION(tea_leaf_dpcg_prolong_Z_device);
}

void TeaOpenCLChunk::tea_leaf_dpcg_subtract_u_kernel
(double * t2_local)
{
    queue.enqueueWriteBuffer(coarse_local_t2, CL_TRUE, 0, 
        local_coarse_x_cells*local_coarse_y_cells*sizeof(double),
        t2_local);

    ENQUEUE_DEFLATION(tea_leaf_dpcg_subtract_u_device);
}

void TeaOpenCLChunk::tea_leaf_dpcg_restrict_zt_kernel
(double * ztr_local)
{
    ENQUEUE_DEFLATION(tea_leaf_dpcg_restrict_ZT_device);

    queue.finish();

    queue.enqueueReadBuffer(coarse_local_ztr, CL_TRUE, 0, 
        local_coarse_x_cells*local_coarse_y_cells*sizeof(double),
        ztr_local);
}

void TeaOpenCLChunk::tea_leaf_dpcg_copy_reduced_t2
(double * global_coarse_t2)
{
    cl::size_t<3> buffer_origin;
    cl::size_t<3> host_origin;
    cl::size_t<3> region;

    size_t buffer_row_pitch;
    size_t host_row_pitch;

    getCoarseCopyParameters(&buffer_origin, &host_origin, &region,
        &buffer_row_pitch, &host_row_pitch);

    // t2 is used as u0 in the coarse solve
    queue.enqueueWriteBufferRect(u0, CL_TRUE,
        buffer_origin,
        host_origin,
        region,
        buffer_row_pitch,
        0,
        host_row_pitch,
        0,
        global_coarse_t2);

    return;
    tea_leaf_calc_residual();
    {
    std::vector<double> result = dumpArray("u0", 0, 0);
    fprintf(stdout, "%d %d\n", chunk_x_cells, chunk_y_cells);
    FILE * chunkout = fopen("chunk.out", "w");
    for (size_t ii = 0; ii < result.size(); ii++)
        fprintf(chunkout, "%.15e ", 1e10*result.at(ii));
    fprintf(chunkout, "\n");
    fclose(chunkout);
    exit(0);
    }
}

void TeaOpenCLChunk::tea_leaf_dpcg_solve_z
(void)
{
    ENQUEUE(tea_leaf_dpcg_solve_z_device);
}

void TeaOpenCLChunk::tea_leaf_dpcg_matmul_zta_kernel
(double * ztaz_local)
{
    ENQUEUE_DEFLATION(tea_leaf_dpcg_matmul_ZTA_device);

    queue.finish();

    queue.enqueueReadBuffer(coarse_local_ztaz, CL_TRUE, 0, 
        local_coarse_x_cells*local_coarse_y_cells*sizeof(double),
        ztaz_local);
}

void TeaOpenCLChunk::tea_leaf_dpcg_init_p_kernel
(void)
{
    ENQUEUE(tea_leaf_dpcg_init_p_device);
}

void TeaOpenCLChunk::tea_leaf_dpcg_store_r_kernel
(void)
{
    ENQUEUE(tea_leaf_dpcg_store_r_device);
}

void TeaOpenCLChunk::tea_leaf_dpcg_calc_rrn_kernel
(double * rrn)
{
    ENQUEUE(tea_leaf_dpcg_calc_rrn_device);

    *rrn = reduceValue<double>(sum_red_kernels_double, reduce_buf_5);
}

void TeaOpenCLChunk::tea_leaf_dpcg_calc_p_kernel
(double beta)
{
    tea_leaf_dpcg_calc_p_device.setArg(3, beta);

    ENQUEUE(tea_leaf_dpcg_calc_p_device);
}

void TeaOpenCLChunk::tea_leaf_dpcg_local_solve
(double * coarse_solve_eps,
 int    * coarse_solve_max_iters,
 int    * it_count,
 double * theta,
 int    * inner_use_ppcg,
 double * inner_cg_alphas,
 double * inner_cg_betas,
 double * inner_ch_alphas,
 double * inner_ch_betas,
 double * t2_result)
{
    cl::size_t<3> buffer_origin;
    cl::size_t<3> host_origin;
    cl::size_t<3> region;

    size_t buffer_row_pitch;
    size_t host_row_pitch;

    getCoarseCopyParameters(&buffer_origin, &host_origin, &region,
        &buffer_row_pitch, &host_row_pitch);

    // t2 is 0 here - we want a 0 initial guess
    queue.enqueueWriteBufferRect(u, CL_TRUE,
        buffer_origin,
        host_origin,
        region,
        buffer_row_pitch,
        0,
        host_row_pitch,
        0,
        t2_result);

    double rro, rrn, pw;

    tea_leaf_calc_residual();
    tea_leaf_cg_init_kernel(&rro);

    double initial = rro;

    rrn = 1e10;

    fprintf(stdout, "before: %e\n", rro);

    for (int ii = 0; (ii < (*coarse_solve_max_iters)) && (sqrt(fabs(rrn)) > (*coarse_solve_eps)*initial); ii++)
    {
        // TODO redo these so it doesnt copy back memory repeatedly
        tea_leaf_cg_calc_w_kernel(&pw);

        double alpha = rro/pw;

        tea_leaf_cg_calc_ur_kernel(alpha, &rrn);

        double beta = rrn/rro;

        tea_leaf_cg_calc_p_kernel(beta);

        rro = rrn;

        inner_cg_alphas[ii] = alpha;
        inner_cg_betas[ii] = beta;

        *it_count = ii + 1;
    }

    fprintf(stdout, "after: %e\n", rrn);
    fprintf(stdout, "%d iters\n", *it_count);
    fprintf(stdout, "\n");

    if (*inner_use_ppcg)
    {
    std::vector<double> result = dumpArray("u", 0, 0);
    fprintf(stdout, "%d %d\n", chunk_x_cells, chunk_y_cells);
    FILE * chunkout = fopen("chunk.out", "w");
    for (size_t ii = 0; ii < result.size(); ii++)
        fprintf(chunkout, "%e ", result.at(ii));
    fprintf(chunkout, "\n");
    fclose(chunkout);
    exit(0);
    }

    // copy back result into t2
    queue.enqueueReadBufferRect(u, CL_TRUE,
        buffer_origin,
        host_origin,
        region,
        buffer_row_pitch,
        0,
        host_row_pitch,
        0,
        t2_result);
}

