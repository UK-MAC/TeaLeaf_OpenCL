#ifndef __CL_TYPE_HDR
#define __CL_TYPE_HDR

#include "CL/cl.hpp"

#include <cstdio>
#include <cstdlib>
#include <map>

#include "kernel_files/definitions.hpp"

#define CL_SAFE_CALL(x) try{x}catch(cl::Error e){DIE("%d %s - %d %s", __LINE__, __FILE__, e.err(), e.what());}

// this function gets called when something goes wrong
#define DIE(...) cloverDie(__LINE__, __FILE__, __VA_ARGS__)

typedef struct cell_info_struct {
    cl_int x_extra;
    cl_int y_extra;
    cl_int x_invert;
    cl_int y_invert;
    cl_int x_face;
    cl_int y_face;
    cl_int grid_type;
} cell_info_t;

// specific sizes and launch offsets for different kernels
typedef struct {
    cl::NDRange global;
    cl::NDRange offset;
} launch_specs_t;

typedef struct kernel_info_struct {
    cl_int x_min;
    cl_int x_max;
    cl_int y_min;
    cl_int y_max;
    cl_int halo_depth;
    cl_int preconditioner_type;
    cl_int x_offset;
    cl_int y_offset;

    cl_int kernel_x_min;
    cl_int kernel_x_max;
    cl_int kernel_y_min;
    cl_int kernel_y_max;
} kernel_info_t;

// reductions
typedef struct {
    cl::Kernel kernel;
    cl::NDRange global_size;
    cl::NDRange local_size;
} reduce_kernel_info_t;

typedef struct {
    // if profiling
    bool profiler_on;
    // type of preconditioner
    int preconditioner_type;
    // which solver to use, enumerated
    int tea_solver;
    // total number of cells in this MPI rank
    size_t x_cells;
    size_t y_cells;
    // halo size
    size_t halo_exchange_depth;
} run_params_t;

// vectors of kernels and work group sizes for a specific reduction
typedef std::vector<reduce_kernel_info_t> reduce_info_vec_t;

class TeaCLTile
{
    friend class TeaCLContext;
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

    // for recording times if profiling is on
    std::map<std::string, double> kernel_times;
    // recording number of times each kernel was called
    std::map<std::string, int> kernel_calls;

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
        tile->enqueueKernel(tile->knl, __LINE__, __FILE__,  \
                      tile->launch_specs.at(#knl).offset,   \
                      tile->launch_specs.at(#knl).global,   \
                      tile->local_size);

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

    // initialise buffers for device
    void initBuffers
    (void);
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

    void packUnpackAllBuffers
    (int fields[NUM_FIELDS], int offsets[NUM_FIELDS], int depth,
     int face, int pack, double * buffer);

    std::map<std::string, cl::Program> built_programs;

    std::map<std::string, launch_specs_t> launch_specs;

    run_params_t run_params;
public:
    TeaCLTile
    (run_params_t run_params, cl::Context context, cl::Device device);
}; // TeaCLTile

class TeaCLContext
{
private:
    run_params_t run_params;

    // tolerance specified in tea.in
    float tolerance;

    // calculate rx/ry to pass back to fortran
    void calcrxry
    (double dt, double * rx, double * ry);

    // ocl things
    cl::Platform platform;
    cl::Context context;

    // mpi rank
    int rank;

    // Where to send debug output
    FILE* DBGOUT;

    // number of tiles
    size_t n_tiles;
    std::vector<TeaCLTile> tiles;
    std::vector<TeaCLTile>::iterator typedef tileit;

    #define FOR_EACH_TILE \
        for (tileit tile = tiles.begin(); tile < tiles.end(); tile++)

    /*
     *  initialisation subroutines
     */

    // initialise context, queue, etc
    void initOcl
    (void);
    // initialise all program stuff, kernels, etc
    void initProgram
    (void);
    // intialise local/global sizes
    void initSizes
    (void);
    // initialise buffers for device
    void initBuffers
    (void);
    // initialise all the arguments for each kernel
    void initArgs
    (void);
    // create reduction kernels
    void initReduction
    (void);

public:
    void field_summary_kernel(double* vol, double* mass,
        double* ie, double* temp);

    void generate_chunk_kernel(const int number_of_states, 
        const double* state_density, const double* state_energy,
        const double* state_xmin, const double* state_xmax,
        const double* state_ymin, const double* state_ymax,
        const double* state_radius, const int* state_geometry,
        const int g_rect, const int g_circ, const int g_point);

    void initialise_chunk_kernel(double d_xmin, double d_ymin,
        double d_dx, double d_dy);

    void update_halo_kernel(const int* fields, int depth,
        const int* chunk_neighbours);
    void update_array
    (cl::Buffer& cur_array,
    const cell_info_t& array_type,
    const int* chunk_neighbours,
    int depth);

    void set_field_kernel();

    // Tea leaf
    void tea_leaf_jacobi_solve_kernel
    (double* error);

    void tea_leaf_cg_init_kernel
    (double * rro);
    void tea_leaf_cg_calc_w_kernel
    (double* pw);
    void tea_leaf_cg_calc_ur_kernel
    (double alpha, double* rrn);
    void tea_leaf_cg_calc_p_kernel
    (double beta);

    void tea_leaf_dpcg_coarsen_matrix_kernel
    (double * Kx_local, double * Ky_local);
    void tea_leaf_dpcg_prolong_z_kernel
    (double * t2_local);
    void tea_leaf_dpcg_subtract_u_kernel
    (double * t2_local);
    void tea_leaf_dpcg_restrict_zt_kernel
    (double * ztr_local);
    void tea_leaf_dpcg_matmul_zta_kernel
    (double * ztaz_local);
    void tea_leaf_dpcg_init_p_kernel
    (void);
    void tea_leaf_dpcg_store_r_kernel
    (void);
    void tea_leaf_dpcg_calc_rrn_kernel
    (double * rrn);
    void tea_leaf_dpcg_calc_p_kernel
    (void);

    void tea_leaf_cheby_init_kernel
    (const double * ch_alphas, const double * ch_betas, int n_coefs,
     const double rx, const double ry, const double theta);
    void tea_leaf_cheby_iterate_kernel
    (const int cheby_calc_steps);

    void ppcg_init
    (const double * ch_alphas, const double * ch_betas,
    const double theta, const int n);
    void ppcg_init_sd_kernel
    (void);
    void tea_leaf_ppcg_inner_kernel
    (int, int, const int*);

    void tea_leaf_finalise();
    void tea_leaf_calc_residual(void);
    void tea_leaf_common_init
    (int coefficient, double dt, double * rx, double * ry,
     int * zero_boundary, int reflective_boundary);
    void tea_leaf_calc_2norm_kernel
    (int norm_array, double* norm);

    void print_profiling_info
    (void);

    void initialise(void);

    void packUnpackAllBuffers
    (int fields[NUM_FIELDS], int offsets[NUM_FIELDS], int depth,
     int face, int pack, double * buffer);
};

class KernelCompileError : std::exception
{
private:
    const std::string _what;
    const int _err;
public:
    KernelCompileError(const char* what, int err):_what(what),_err(err){}
    ~KernelCompileError() throw(){}

    const char* what() const throw() {return this->_what.c_str();}

    int err() const throw() {return this->_err;}
};

#endif
