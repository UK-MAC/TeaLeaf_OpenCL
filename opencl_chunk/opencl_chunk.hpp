#ifndef __TEA_OPENCL_CHUNK_HEADER
#define __TEA_OPENCL_CHUNK_HEADER

#include <map>
#include "../ctx_common.hpp"

#include "CL/cl.hpp"

#define CL_SAFE_CALL(x) try{x}catch(cl::Error e){DIE("%d %s - %d %s", __LINE__, __FILE__, e.err(), e.what());}

class TeaOpenCLChunk : public TeaChunk
{
private:
    // kernels
    cl::Kernel set_field_device;
    cl::Kernel field_summary_device;

    cl::Kernel generate_chunk_device;
    cl::Kernel generate_chunk_init_device;
    cl::Kernel generate_chunk_init_u_device;

    cl::Kernel initialise_chunk_first_device;
    cl::Kernel initialise_chunk_second_device;

    // halo updates
    cl::Kernel update_halo_top_device;
    cl::Kernel update_halo_bottom_device;
    cl::Kernel update_halo_left_device;
    cl::Kernel update_halo_right_device;
    // mpi packing
    cl::Kernel pack_left_buffer_device;
    cl::Kernel unpack_left_buffer_device;
    cl::Kernel pack_right_buffer_device;
    cl::Kernel unpack_right_buffer_device;
    cl::Kernel pack_bottom_buffer_device;
    cl::Kernel unpack_bottom_buffer_device;
    cl::Kernel pack_top_buffer_device;
    cl::Kernel unpack_top_buffer_device;

    // main buffers, with sub buffers for each offset
    cl::Buffer left_buffer;
    cl::Buffer right_buffer;
    cl::Buffer bottom_buffer;
    cl::Buffer top_buffer;

    // jacobi solver
    cl::Kernel tea_leaf_jacobi_copy_u_device;
    cl::Kernel tea_leaf_jacobi_solve_device;

    // cg solver
    cl::Kernel tea_leaf_cg_solve_init_p_device;
    cl::Kernel tea_leaf_cg_solve_calc_w_device;
    cl::Kernel tea_leaf_cg_solve_calc_ur_device;
    cl::Kernel tea_leaf_cg_solve_calc_rrn_device;
    cl::Kernel tea_leaf_cg_solve_calc_p_device;

    // chebyshev solver
    cl::Kernel tea_leaf_cheby_solve_init_p_device;
    cl::Kernel tea_leaf_cheby_solve_calc_u_device;
    cl::Kernel tea_leaf_cheby_solve_calc_p_device;
    cl::Kernel tea_leaf_calc_2norm_device;

    // ppcg solver
    cl::Kernel tea_leaf_ppcg_solve_init_sd_device;
    cl::Kernel tea_leaf_ppcg_solve_calc_sd_device;
    cl::Kernel tea_leaf_ppcg_solve_update_r_device;

    // deflated CG solver
    cl::Kernel tea_leaf_dpcg_coarsen_matrix_device;
    cl::Kernel tea_leaf_dpcg_prolong_Z_device;
    cl::Kernel tea_leaf_dpcg_subtract_u_device;
    cl::Kernel tea_leaf_dpcg_restrict_ZT_device;
    cl::Kernel tea_leaf_dpcg_matmul_ZTA_device;
    cl::Kernel tea_leaf_dpcg_init_p_device;
    cl::Kernel tea_leaf_dpcg_store_r_device;
    cl::Kernel tea_leaf_dpcg_calc_rrn_device;
    cl::Kernel tea_leaf_dpcg_calc_p_device;
    //cl::Kernel tea_leaf_dpcg_calc_zrnorm_device;
    cl::Kernel tea_leaf_dpcg_solve_z_device;

    // used to hold the alphas/beta used in chebyshev solver - different from CG ones!
    cl::Buffer ch_alphas_device, ch_betas_device;

    // preconditioner related
    cl::Kernel tea_leaf_block_init_device;
    cl::Kernel tea_leaf_block_solve_device;
    cl::Kernel tea_leaf_init_jac_diag_device;

    // common
    cl::Kernel tea_leaf_finalise_device;
    cl::Kernel tea_leaf_calc_residual_device;
    cl::Kernel tea_leaf_init_common_device;
    cl::Kernel tea_leaf_zero_boundary_device;

    // buffers
    cl::Buffer density;
    cl::Buffer energy0;
    cl::Buffer energy1;
    cl::Buffer volume;

    cl::Buffer cellx;
    cl::Buffer celly;
    cl::Buffer celldx;
    cl::Buffer celldy;
    cl::Buffer vertexx;
    cl::Buffer vertexy;
    cl::Buffer vertexdx;
    cl::Buffer vertexdy;

    cl::Buffer xarea;
    cl::Buffer yarea;

    cl::Buffer vector_p;
    cl::Buffer vector_r;
    cl::Buffer vector_w;
    cl::Buffer vector_Mi;
    cl::Buffer vector_Kx;
    cl::Buffer vector_Ky;
    cl::Buffer vector_sd;

    // for reduction in field_summary
    cl::Buffer reduce_buf_1;
    cl::Buffer reduce_buf_2;
    cl::Buffer reduce_buf_3;
    cl::Buffer reduce_buf_4;
    cl::Buffer reduce_buf_5;
    cl::Buffer reduce_buf_6;

    cl::Buffer cp, bfp;

    cl::Buffer u, u0;
    cl::Buffer vector_z;
    cl::Buffer vector_rm1;

    // small arrays that are the size of the number of work groups launched
    // could reuse them but they're tiny
    cl::Buffer coarse_local_Kx;
    cl::Buffer coarse_local_Ky;
    cl::Buffer coarse_local_t2;
    cl::Buffer coarse_local_ztr;
    cl::Buffer coarse_local_ztaz;

    cl::Device device;
    cl::Context context;
    cl::CommandQueue queue;

    std::string device_type_prepro;

    // global size for kernels
    cl::NDRange global_size;
    cl::NDRange local_size;

    reduce_info_vec_t sum_red_kernels_double;

    // compile a file and the contained kernels, and check for errors
    void compileKernel
    (std::stringstream& options,
     const std::string& source_name,
     const char* kernel_name,
     cl::Kernel& kernel,
     int launch_x_min, int launch_x_max,
     int launch_y_min, int launch_y_max);

    // number of cells reduced
    int reduced_cells;

    // sizes for launching update halo kernels - l/r and u/d updates
    std::map<int, cl::NDRange> update_lr_global_size;
    std::map<int, cl::NDRange> update_bt_global_size;
    std::map<int, cl::NDRange> update_lr_local_size;
    std::map<int, cl::NDRange> update_bt_local_size;
    std::map<int, cl::NDRange> update_lr_offset;
    std::map<int, cl::NDRange> update_bt_offset;

    std::vector<double> dumpArray
    (const std::string& arr_name, int x_extra, int y_extra);
    std::map<std::string, cl::Buffer> arr_names;

    // enqueue a kernel
    void enqueueKernel
    (cl::Kernel const& kernel,
     int line, const char* file,
     const cl::NDRange offset,
     const cl::NDRange global_range,
     const cl::NDRange local_range,
     const std::vector< cl::Event > * const events=NULL,
     cl::Event * const event=NULL);

    #define ENQUEUE(knl)                                \
        enqueueKernel(knl, __LINE__, __FILE__,          \
                      launch_specs.at(#knl).offset,     \
                      launch_specs.at(#knl).global,     \
                      local_size);

    // always uses SUB_TILE_BLOCK_SIZE as local size
    #define ENQUEUE_DEFLATION(knl)                      \
        enqueueKernel(knl, __LINE__, __FILE__,          \
                      launch_specs.at(#knl).offset,     \
                      launch_specs.at(#knl).global,     \
                      cl::NDRange(SUB_TILE_BLOCK_SIZE, SUB_TILE_BLOCK_SIZE));

    template <typename T>
    T reduceValue
    (reduce_info_vec_t& red_kernels,
     const cl::Buffer& results_buf);
     //T* result, cl::Event* copy_event, cl::Event*);

    void initTileQueue
    (void);

    launch_specs_t findPaddingSize
    (int vmin, int vmax, int hmin, int hmax);

    cl::Program compileProgram
    (const std::string& source,
     const std::string& options);

    // create reduction kernels
    void initReduction
    (void);
    // initialise all program stuff, kernels, etc
    void initProgram
    (void);
    void initMemory
    (void);
    // initialise all the arguments for each kernel
    void initArgs
    (void);
    void initSizes
    (void);

    void update_array
    (cl::Buffer& cur_array,
     const cell_info_t& array_type,
     const int* chunk_neighbours,
     int depth);

    void tea_leaf_calc_2norm_set_vector
    (int norm_array);

    std::map<std::string, cl::Program> built_programs;

    std::map<std::string, launch_specs_t> launch_specs;

    run_params_t run_params;

    void getCoarseCopyParameters
    (cl::size_t<3> * buffer_origin,
     cl::size_t<3> * host_origin,
     cl::size_t<3> * region,
     size_t * buffer_row_pitch,
     size_t * host_row_pitch);

public:
    TeaOpenCLChunk
    (run_params_t run_params, cl::Context context, cl::Device device,
     int x_cells, int y_cells, int coarse_x_cells, int coarse_y_cells);

    virtual void packUnpackAllBuffers
    (int fields[NUM_FIELDS], int offsets[NUM_FIELDS], int depth,
     int face, int pack, double * buffer);

    virtual void set_field_kernel
    (void);

    virtual void field_summary_kernel
    (double* vol, double* mass, double* ie, double* temp);

    virtual void generate_chunk_kernel
    (const int number_of_states, 
    const double* state_density, const double* state_energy,
    const double* state_xmin, const double* state_xmax,
    const double* state_ymin, const double* state_ymax,
    const double* state_radius, const int* state_geometry,
    const int g_rect, const int g_circ, const int g_point);

    virtual void update_halo_kernel
    (const int* chunk_neighbours,
     const int* fields,
     const int depth);

    virtual void initialise_chunk_kernel
    (double d_xmin, double d_ymin, double d_dx, double d_dy);

    virtual void calcrxry
    (double dt, double * rx, double * ry);

    virtual void tea_leaf_calc_2norm_kernel
    (int norm_array, double* norm);

    virtual void tea_leaf_common_init
    (int coefficient, double dt, double * rx, double * ry,
     int * zero_boundary, int reflective_boundary);

    virtual void tea_leaf_finalise
    (void);

    virtual void tea_leaf_calc_residual
    (void);

    virtual void tea_leaf_cg_init_kernel
    (double * rro);

    virtual void tea_leaf_cg_calc_w_kernel
    (double* pw);

    virtual void tea_leaf_cg_calc_ur_kernel
    (double alpha, double* rrn);

    virtual void tea_leaf_cg_calc_p_kernel
    (double beta);

    virtual void tea_leaf_cheby_init_kernel
    (const double * ch_alphas, const double * ch_betas, int n_coefs,
     const double rx, const double ry, const double theta);

    virtual void tea_leaf_cheby_iterate_kernel
    (const int cheby_calc_step);

    virtual void tea_leaf_jacobi_solve_kernel
    (double* error);

    virtual void ppcg_init
    (const double * ch_alphas, const double * ch_betas,
     const double theta, const int n_inner_steps);

    virtual void ppcg_init_sd_kernel
    (void);

    virtual void tea_leaf_ppcg_inner_kernel
    (int inner_step, int bounds_extra, const int* chunk_neighbours);

    virtual void tea_leaf_dpcg_prolong_z_kernel
    (double * t2_local);

    virtual void tea_leaf_dpcg_subtract_u_kernel
    (double * t2_local);

    virtual void tea_leaf_dpcg_restrict_zt_kernel
    (double * ztr_local);

    virtual void tea_leaf_dpcg_solve_z
    (void);

    virtual void tea_leaf_dpcg_matmul_zta_kernel
    (double * ztaz_local);

    virtual void tea_leaf_dpcg_init_p_kernel
    (void);

    virtual void tea_leaf_dpcg_store_r_kernel
    (void);

    virtual void tea_leaf_dpcg_calc_rrn_kernel
    (double * rrn);

    virtual void tea_leaf_dpcg_calc_p_kernel
    (double beta);

    virtual void tea_leaf_dpcg_coarsen_matrix_kernel
    (double * host_Kx, double * host_Ky);

    virtual void tea_leaf_dpcg_copy_reduced_coarse_grid
    (double * global_coarse_Kx, double * global_coarse_Ky, double * global_coarse_Di);

    virtual void tea_leaf_dpcg_copy_reduced_t2
    (double * global_coarse_t2);

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
     double * t2_result);
}; // TeaOpenCLChunk

#endif

