#ifndef __CL_TYPE_HDR
#define __CL_TYPE_HDR

#include <cstdio>
#include <cstdlib>
#include <map>

#include "CL/cl.hpp"

#include "kernel_files/definitions.hpp"

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

/*
 *  TODO
 *  Hardcoded to have 2 chunk levels (coarse and fine). When properly merged with
 *  the Fortran then it should technically be possible to have an arbitrary
 *  layering of chunks at different coarse levels, corresponding to the 'chunk'
 *  value
 */
const static int fine_chunk = 0;
const static int coarse_chunk = 1;

class TeaOpenCLChunk;

#if __cplusplus > 199711L
#include <memory>
typedef std::shared_ptr<TeaOpenCLChunk> chunk_ptr_t;
#else
#include <tr1/memory>
typedef std::tr1::shared_ptr<TeaOpenCLChunk> chunk_ptr_t;
#endif

#include "opencl_chunk/opencl_chunk.hpp"

// TODO
//#include "fortran_chunk/fortran_chunk.hpp"

class TeaCLContext
{
private:
    run_params_t run_params;

    // tolerance specified in tea.in
    float tolerance;

    // calculate rx/ry to pass back to fortran
    void calcrxry
    (double dt, double * rx, double * ry);

    // mpi rank
    int rank;

    // Where to send debug output
    FILE* DBGOUT;

    // for OpenCL
    cl::Context context;

    // number of chunks
    size_t n_chunks;
    std::map<int, chunk_ptr_t> chunks;
    std::map<int, chunk_ptr_t>::iterator typedef chunkit_t;

    #define FOR_EACH_CHUNK \
        for (chunkit_t chunk_it = chunks.begin(); chunk_it != chunks.end(); chunk_it++)

    /*
     *  initialisation subroutines
     */

    // initialise context, queue, etc
    void initOcl
    (int * tile_sizes, int n_tiles);
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
    void tea_leaf_dpcg_copy_reduced_coarse_grid
    (double * global_coarse_Kx, double * global_coarse_Ky, double * global_coarse_Di);
    void tea_leaf_dpcg_copy_reduced_t2
    (double * global_coarse_t2);
    void tea_leaf_dpcg_prolong_z_kernel
    (double * t2_local);
    void tea_leaf_dpcg_subtract_u_kernel
    (double * t2_local);
    void tea_leaf_dpcg_restrict_zt_kernel
    (double * ztr_local);
    void tea_leaf_dpcg_solve_z_kernel
    (void);
    void tea_leaf_dpcg_matmul_zta_kernel
    (double * ztaz_local);
    void tea_leaf_dpcg_init_p_kernel
    (void);
    void tea_leaf_dpcg_store_r_kernel
    (void);
    void tea_leaf_dpcg_calc_rrn_kernel
    (double * rrn);
    void tea_leaf_dpcg_calc_p_kernel
    (double beta);

    void tea_leaf_dpcg_local_solve
    (double   coarse_solve_eps,
     int      coarse_solve_max_iters,
     int    * it_count,
     double   theta,
     int      inner_use_ppcg,
     int      ppcg_max_iters,
     double * inner_cg_alphas,
     double * inner_cg_betas,
     double * inner_ch_alphas,
     double * inner_ch_betas,
     double * t2_result);

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

    void packUnpackAllBuffers
    (int fields[NUM_FIELDS], int offsets[NUM_FIELDS], int depth,
     int face, int pack, double * buffer);

    void initialise
    (int * tile_sizes, int n_tiles);
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
