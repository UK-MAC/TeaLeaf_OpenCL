#include "ocl_common.hpp"

extern CloverChunk chunk;

extern "C" void tea_leaf_kernel_init_ocl_
(int *x_min,int *x_max,int *y_min,int *y_max,
 const double * celldx,
 const double * celldy,
 const double * volume,
 const double * density1,
 const double * energy1,
 const double * work_array1,
 const double * u,
 const double * work_array2,
 const double * work_array3,
 const double * work_array4,
 const double * work_array5,
 const double * work_array6,
 const double * work_array7,
 const int    * coefficient,
       double * dt,
       double * rx,
       double * ry)
{
    chunk.tea_leaf_init(*coefficient, *dt, rx, ry);
}

extern "C" void tea_leaf_kernel_solve_ocl_
(int *x_min,int *x_max,int *y_min,int *y_max,
 const double * rx,
 const double * ry,
 const double * work_array6,
 const double * work_array7,
       double * error,
 const double * work_array1,
 const double * u,
 const double * work_array2)
{
    chunk.tea_leaf_kernel(*rx, *ry, error);
}

extern "C" void tea_leaf_kernel_finalise_ocl_
(int *x_min,int *x_max,int *y_min,int *y_max,
 const double * rx,
 const double * ry,
 const double * density1,
 const double * energy1,
 const double * u)
{
    chunk.tea_leaf_finalise();
}

#define CONDUCTIVITY 1
#define RECIP_CONDUCTIVITY 2

void CloverChunk::calcrxry
(double dt, double * rx, double * ry)
{
    static int initd = 0;
    if (!initd)
    {
        // make sure intialise chunk has finished
        queue.finish();
        // celldx doesnt change after that so check once
        initd = 1;
    }

    double dx, dy;

    try
    {
        // celldx/celldy never change, but done for consistency with fortran
        queue.enqueueReadBuffer(celldx, CL_TRUE,
            sizeof(double)*x_min, sizeof(double), &dx);
        queue.enqueueReadBuffer(celldy, CL_TRUE,
            sizeof(double)*y_min, sizeof(double), &dy);
    }
    catch (cl::Error e)
    {
        DIE("Error in copying back value from celldx/celldy (%d - %s)\n",
            e.err(), e.what());
    }

    *rx = dt/(dx*dx);
    *ry = dt/(dy*dy);
}

void CloverChunk::tea_leaf_init
(int coefficient, double dt, double * rx, double * ry)
{
    if (coefficient != CONDUCTIVITY && coefficient != RECIP_CONDUCTIVITY)
    {
        DIE("Unknown coefficient %d passed to tea leaf\n", coefficient);
    }

    // copy back dx/dy and calculate rx/ry
    calcrxry(dt, rx, ry);

#if defined(TL_USE_CG)
    // copy u, get density value modified by coefficient
    tea_leaf_cg_init_u_device.setArg(6, coefficient);
    ENQUEUE(tea_leaf_cg_init_u_device);

    // init Kx, Ky
    ENQUEUE(tea_leaf_cg_init_directions_device);

    // get initial guess in w, r, etc
    tea_leaf_cg_init_others_device.setArg(9, *rx);
    tea_leaf_cg_init_others_device.setArg(10, *ry);
    ENQUEUE(tea_leaf_cg_init_others_device);

    // stop it copying back which wastes time
    double bb = reduceValue<double>(sum_red_kernels_double, reduce_buf_1, true);
    double rro = reduceValue<double>(sum_red_kernels_double, reduce_buf_2);

    // only needs to be set once
    tea_leaf_cg_solve_calc_w_device.setArg(5, *rx);
    tea_leaf_cg_solve_calc_w_device.setArg(6, *ry);

    // initialise rro
    tea_leaf_cg_solve_calc_p_device.setArg(0, rro);
    tea_leaf_cg_solve_calc_ur_device.setArg(0, rro);
#else
    tea_leaf_jacobi_init_device.setArg(6, coefficient);
    ENQUEUE(tea_leaf_jacobi_init_device);

    tea_leaf_jacobi_solve_device.setArg(0, *rx);
    tea_leaf_jacobi_solve_device.setArg(1, *ry);
#endif
}

void CloverChunk::tea_leaf_kernel
(double rx, double ry, double* error)
{
#if defined(TL_USE_CG)
    ENQUEUE(tea_leaf_cg_solve_calc_w_device);
    double pw = reduceValue<double>(sum_red_kernels_double, reduce_buf_3, true);

    ENQUEUE(tea_leaf_cg_solve_calc_ur_device);
    double rrn = reduceValue<double>(sum_red_kernels_double, reduce_buf_4);

    ENQUEUE(tea_leaf_cg_solve_calc_p_device);

    // re set rro to rrn
    tea_leaf_cg_solve_calc_p_device.setArg(0, rrn);
    tea_leaf_cg_solve_calc_ur_device.setArg(0, rrn);

    *error = rrn;
#else
    ENQUEUE(tea_leaf_jacobi_copy_u_device);
    ENQUEUE(tea_leaf_jacobi_solve_device);

    *error = reduceValue<double>(max_red_kernels_double, reduce_buf_1);
#endif
}

void CloverChunk::tea_leaf_finalise
(void)
{
    ENQUEUE(tea_leaf_finalise_device);
}

