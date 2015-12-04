#ifndef __GENERIC_CHUNK_HDR
#define __GENERIC_CHUNK_HDR

class TeaChunk
{
public:
    // for recording times if profiling is on
    std::map<std::string, double> kernel_times;
    // recording number of times each kernel was called
    std::map<std::string, int> kernel_calls;

    // number of cells
    const int chunk_x_cells;
    const int chunk_y_cells;
    // size of local portion of next coarsest grid
    const int local_coarse_x_cells;
    const int local_coarse_y_cells;

    // mpi rank
    int rank;

    virtual void packUnpackAllBuffers
    (int fields[NUM_FIELDS], int offsets[NUM_FIELDS], int depth,
     int face, int pack, double * buffer)=0;

    virtual void set_field_kernel
    (void)=0;

    virtual void field_summary_kernel
    (double* vol, double* mass, double* ie, double* temp)=0;

    virtual void generate_chunk_kernel
    (const int number_of_states, 
    const double* state_density, const double* state_energy,
    const double* state_xmin, const double* state_xmax,
    const double* state_ymin, const double* state_ymax,
    const double* state_radius, const int* state_geometry,
    const int g_rect, const int g_circ, const int g_point)=0;

    virtual void update_halo_kernel
    (const int* chunk_neighbours,
     const int* fields,
     const int depth)=0;

    virtual void initialise_chunk_kernel
    (double d_xmin, double d_ymin, double d_dx, double d_dy)=0;

    virtual void calcrxry
    (double dt, double * rx, double * ry)=0;

    virtual void tea_leaf_calc_2norm_kernel
    (int norm_array, double* norm)=0;

    virtual void tea_leaf_common_init
    (int coefficient, double dt, double * rx, double * ry,
     int * zero_boundary, int reflective_boundary)=0;

    virtual void tea_leaf_finalise
    (void)=0;

    virtual void tea_leaf_calc_residual
    (void)=0;

    virtual void tea_leaf_cg_init_kernel
    (double * rro)=0;

    virtual void tea_leaf_cg_calc_w_kernel
    (double* pw)=0;

    virtual void tea_leaf_cg_calc_ur_kernel
    (double alpha, double* rrn)=0;

    virtual void tea_leaf_cg_calc_p_kernel
    (double beta)=0;

    virtual void tea_leaf_cheby_init_kernel
    (const double * ch_alphas, const double * ch_betas, int n_coefs,
     const double rx, const double ry, const double theta)=0;

    virtual void tea_leaf_cheby_iterate_kernel
    (const int cheby_calc_step)=0;

    virtual void tea_leaf_jacobi_solve_kernel
    (double* error)=0;

    virtual void ppcg_init
    (const double * ch_alphas, const double * ch_betas,
     const double theta, const int n_inner_steps)=0;

    virtual void ppcg_init_sd_kernel
    (void)=0;

    virtual void tea_leaf_ppcg_inner_kernel
    (int inner_step, int bounds_extra, const int* chunk_neighbours)=0;

    virtual void tea_leaf_dpcg_prolong_z_kernel
    (double * t2_local)=0;

    virtual void tea_leaf_dpcg_subtract_u_kernel
    (double * t2_local)=0;

    virtual void tea_leaf_dpcg_restrict_zt_kernel
    (double * ztr_local)=0;

    virtual void tea_leaf_dpcg_solve_z
    (void)=0;

    virtual void tea_leaf_dpcg_matmul_zta_kernel
    (double * ztaz_local)=0;

    virtual void tea_leaf_dpcg_init_p_kernel
    (void)=0;

    virtual void tea_leaf_dpcg_store_r_kernel
    (void)=0;

    virtual void tea_leaf_dpcg_calc_rrn_kernel
    (double * rrn)=0;

    virtual void tea_leaf_dpcg_calc_p_kernel
    (double beta)=0;

    virtual void tea_leaf_dpcg_coarsen_matrix_kernel
    (double * host_Kx, double * host_Ky)=0;

    virtual void tea_leaf_dpcg_copy_reduced_coarse_grid
    (double * global_coarse_Kx, double * global_coarse_Ky, double * global_coarse_Di)=0;

    virtual void tea_leaf_dpcg_copy_reduced_t2
    (double * global_coarse_t2)=0;

    virtual void tea_leaf_dpcg_local_solve
    (double * coarse_solve_eps,
     int    * coarse_solve_max_iters,
     int    * it_count,
     double * theta,
     int    * inner_use_ppcg,
     double * inner_cg_alphas,
     double * inner_cg_betas,
     double * inner_ch_alphas,
     double * inner_ch_betas,
     double * t2_result)=0;

    TeaChunk
    (int x_cells, int y_cells, int coarse_x_cells, int coarse_y_cells);
}; // TeaChunk

#endif

