#ifndef __TEA_OPENCL_TILE_HEADER
#define __TEA_OPENCL_TILE_HEADER

#include "CL/cl.hpp"

#define CL_SAFE_CALL(x) try{x}catch(cl::Error e){DIE("%d %s - %d %s", __LINE__, __FILE__, e.err(), e.what());}

class TeaOpenCLTile : public TeaTile
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

    // dpcg coarse grids
    cl::Buffer Kx_coarse;
    cl::Buffer Ky_coarse;
    cl::Buffer Di_coarse;
    cl::Buffer t1_coarse;
    cl::Buffer t2_coarse;

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

    // number of cells
    int tile_x_cells;
    int tile_y_cells;

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

    #define ENQUEUE(knl)                                    \
        enqueueKernel(knl, __LINE__, __FILE__,  \
                      launch_specs.at(#knl).offset,   \
                      launch_specs.at(#knl).global,   \
                      local_size);

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
public:
    TeaOpenCLTile
    (run_params_t run_params, cl::Context context, cl::Device device);
}; // TeaOpenCLTile

#endif

