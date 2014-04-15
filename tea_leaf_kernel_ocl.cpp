#include "ocl_common.hpp"

extern CloverChunk chunk;

// same as in fortran
#define CONDUCTIVITY 1
#define RECIP_CONDUCTIVITY 2

// jacobi solver functions
extern "C" void tea_leaf_kernel_init_ocl_
(const int * coefficient, double * dt, double * rx, double * ry)
{
    chunk.tea_leaf_init_jacobi(*coefficient, *dt, rx, ry);
}

extern "C" void tea_leaf_kernel_solve_ocl_
(const double * rx, const double * ry, double * error)
{
    chunk.tea_leaf_kernel_jacobi(*rx, *ry, error);
}

// CG solver functions
extern "C" void tea_leaf_kernel_init_cg_ocl_
(const int * coefficient, double * dt, double * rx, double * ry, double * rro)
{
    chunk.tea_leaf_init_cg(*coefficient, *dt, rx, ry, rro);
}

extern "C" void tea_leaf_kernel_solve_cg_ocl_calc_w_
(const double * rx, const double * ry, double * pw)
{
    chunk.tea_leaf_kernel_cg_calc_w(*rx, *ry, pw);
}
extern "C" void tea_leaf_kernel_solve_cg_ocl_calc_ur_
(double * alpha, double * rrn)
{
    chunk.tea_leaf_kernel_cg_calc_ur(*alpha, rrn);
}
extern "C" void tea_leaf_kernel_solve_cg_ocl_calc_p_
(double * beta)
{
    chunk.tea_leaf_kernel_cg_calc_p(*beta);
}

// used by both
extern "C" void tea_leaf_kernel_finalise_ocl_
(void)
{
    chunk.tea_leaf_finalise();
}

// copy back dx/dy and calculate rx/ry
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

/********************/
#include <cassert>

void CloverChunk::tea_leaf_init_cg
(int coefficient, double dt, double * rx, double * ry, double * rro)
{
    if (coefficient != CONDUCTIVITY && coefficient != RECIP_CONDUCTIVITY)
    {
        DIE("Unknown coefficient %d passed to tea leaf\n", coefficient);
    }

    assert(tea_solver == TEA_ENUM_CG || tea_solver == TEA_ENUM_CHEBYSHEV);

    calcrxry(dt, rx, ry);

    // only needs to be set once
    tea_leaf_cg_solve_calc_w_device.setArg(5, *rx);
    tea_leaf_cg_solve_calc_w_device.setArg(6, *ry);
    tea_leaf_cg_init_others_device.setArg(8, *rx);
    tea_leaf_cg_init_others_device.setArg(9, *ry);
    tea_leaf_init_diag_device.setArg(3, *rx);
    tea_leaf_init_diag_device.setArg(4, *ry);

    // copy u, get density value modified by coefficient
    tea_leaf_cg_init_u_device.setArg(6, coefficient);
    //ENQUEUE(tea_leaf_cg_init_u_device);
    ENQUEUE_OFFSET(tea_leaf_cg_init_u_device);

    // init Kx, Ky
    //ENQUEUE(tea_leaf_cg_init_directions_device);
    ENQUEUE_OFFSET(tea_leaf_cg_init_directions_device);

    // premultiply Kx/Ky
    //ENQUEUE(tea_leaf_init_diag_device);
    ENQUEUE_OFFSET(tea_leaf_init_diag_device);

    // get initial guess in w, r, etc
    //ENQUEUE(tea_leaf_cg_init_others_device);
    ENQUEUE_OFFSET(tea_leaf_cg_init_others_device);

    if (tea_solver == TEA_ENUM_CHEBYSHEV)
    {
        // copy into u0 for later residual check
        queue.finish();
        queue.enqueueCopyBuffer(u, u0, 0, 0, (x_max+4) * (y_max+4) * sizeof(double));
        cheby_calc_steps = 0;
        cg_alphas.reserve(max_cheby_cg_steps);
        cg_betas.reserve(max_cheby_cg_steps);
        tea_leaf_cheby_solve_calc_p_device.setArg(10, *rx);
        tea_leaf_cheby_solve_calc_p_device.setArg(11, *ry);
    }
    cg_alphas.clear();
    cg_betas.clear();

    *rro = reduceValue<double>(sum_red_kernels_double, reduce_buf_2);
}

void CloverChunk::tea_leaf_kernel_cg_calc_w
(double rx, double ry, double* pw)
{
    if (tea_solver == TEA_ENUM_CHEBYSHEV && cheby_calc_steps > 0)
    {
        return;
    }

    //ENQUEUE(tea_leaf_cg_solve_calc_w_device);
    ENQUEUE_OFFSET(tea_leaf_cg_solve_calc_w_device);
    *pw = reduceValue<double>(sum_red_kernels_double, reduce_buf_3);
}

#include <cmath>
#include <cassert>
#include <limits>
#include <algorithm>
#include <iostream>
static std::pair<double, double> calcEigenvalues
(std::vector<double> alphas,
 std::vector<double> betas)
{
    int order = alphas.size();
    if (order<1) return std::make_pair(0.0, 0.0);

    // calculate eigenvalues
    std::vector<double> diag(order);
    std::vector<double> offdiag(order);
    for (int ii = 0; ii < order; ii++)
    {
        diag[ii] = 1.0/alphas[ii];
        if (ii > 0)
        {
            diag[ii] += (betas[ii-1]/alphas[ii-1]);
        }
        if (ii < order - 1)
        {
            offdiag[ii] = sqrt(betas[ii])/alphas[ii];
        }
    }

    int info;
    tqli_(&diag.front(), &offdiag.front(), &order, NULL, &info);
    std::sort(diag.begin(), diag.end());
    /*
    for (int ii = 0; ii < order; ii++)
    {
        fprintf(stdout, "%f ", diag[ii]);
    }
    fprintf(stdout, "\n");
    */

    assert(!info);

    // calculate alphas/betas
    const double eigmin = diag.front();
    const double eigmax = diag.back();

    return std::make_pair(eigmin, eigmax);

    fprintf(stdout, "Step %d, extreme eigenvalues = [%f %f] (cn = %f)\n",
        order, eigmin, eigmax, eigmax/eigmin);
}

void CloverChunk::tea_leaf_kernel_cg_calc_ur
(double alpha, double* rrn)
{
    if (tea_solver == TEA_ENUM_CHEBYSHEV &&
        (cg_alphas.size() >= max_cheby_cg_steps))
    {
        tea_leaf_cheby_solve_calc_p_device.setArg(12, cheby_calc_steps);

        if (cheby_calc_steps == 0)
        {
            std::pair<double, double> eigs = calcEigenvalues(cg_alphas, cg_betas);
            const double eigmin = eigs.first;
            const double eigmax = eigs.second;

            const double theta = (eigmax + eigmin)/2;
            const double delta = (eigmax - eigmin)/2;
            const double sigma = theta/delta;

            double rho_old = 1.0/sigma;
            // TODO for now, precalculate lots - estimate properly using eigenvalues
            std::vector<double> ch_alphas, ch_betas;
            for (int ii = 0; ii < 1000; ii++)
            {
                const double rho_new = 1.0/(2.0*sigma - rho_old);
                const double ch_alpha = rho_new*rho_old;
                const double ch_beta = 2.0*rho_new/delta;
                rho_old = rho_new;

                ch_alphas.push_back(ch_alpha);
                ch_betas.push_back(ch_beta);
            }

            const size_t ch_buf_sz = ch_alphas.size()*sizeof(double);

            // upload to device
            ch_alphas_device = cl::Buffer(context, CL_MEM_READ_ONLY, ch_buf_sz);
            queue.enqueueWriteBuffer(ch_alphas_device, CL_TRUE, 0,
                ch_buf_sz, &ch_alphas.front());
            ch_betas_device = cl::Buffer(context, CL_MEM_READ_ONLY, ch_buf_sz);
            queue.enqueueWriteBuffer(ch_betas_device, CL_TRUE, 0,
                ch_buf_sz, &ch_betas.front());
            tea_leaf_cheby_solve_calc_p_device.setArg(8, ch_alphas_device);
            tea_leaf_cheby_solve_calc_p_device.setArg(9, ch_betas_device);

            tea_leaf_cheby_solve_loop_calc_u_device.setArg(3, ch_alphas_device);
            tea_leaf_cheby_solve_loop_calc_u_device.setArg(4, ch_betas_device);

            tea_leaf_cheby_solve_init_p_device.setArg(2, theta);

            // this will junk p but we don't need it anyway
            ENQUEUE(tea_leaf_cheby_solve_calc_p_device);
            // then correct p
            ENQUEUE(tea_leaf_cheby_solve_init_p_device);

            // calculate the max number of iterations expected
            double bb;

            // get bb
            tea_leaf_cheby_solve_calc_resid_device.setArg(0, u0);
            ENQUEUE(tea_leaf_cheby_solve_calc_resid_device);
            bb = reduceValue<double>(sum_red_kernels_double, reduce_buf_1);

            // get rrn
            tea_leaf_cheby_solve_calc_resid_device.setArg(0, work_array_2);
            ENQUEUE(tea_leaf_cheby_solve_calc_resid_device);
            *rrn = reduceValue<double>(sum_red_kernels_double, reduce_buf_1);

            const double EPS = std::numeric_limits<double>::epsilon();
            const double it_alpha = tolerance/(4*(*rrn));
            const double cn = eigmax/eigmin;
            const double gamma = (sqrt(cn) - 1)/(sqrt(cn) + 1);
            est_itc = log(it_alpha)/(2*log(gamma));

            fprintf(stdout, "I guess it will take %d iterations\n", est_itc);
            //fprintf(stdout, "Residual going in = %E\n", *rrn);
        }

        if (cheby_calc_steps % 5)
        {
            tea_leaf_cheby_solve_loop_calc_u_device.setArg(5, cheby_calc_steps);
            //ENQUEUE(tea_leaf_cheby_solve_loop_calc_u_device);
            ENQUEUE_OFFSET(tea_leaf_cheby_solve_loop_calc_u_device);
        }
        else
        {
            //ENQUEUE(tea_leaf_cheby_solve_calc_u_device);
            ENQUEUE_OFFSET(tea_leaf_cheby_solve_calc_u_device);
            //ENQUEUE(tea_leaf_cheby_solve_calc_p_device);
            ENQUEUE_OFFSET(tea_leaf_cheby_solve_calc_p_device);
        }

        cheby_calc_steps++;

        // past esitmated number of iterations
        if ((cheby_calc_steps == est_itc)
        // or past estimated and  check evry 5th step afterwards (arbitrary number XXX)
        || (cheby_calc_steps > est_itc && !(cheby_calc_steps % 25)))
        {
            ENQUEUE(tea_leaf_cheby_solve_calc_resid_device);
            *rrn = reduceValue<double>(sum_red_kernels_double, reduce_buf_1);
        }
        else
        {
            // avoid a reduction
            *rrn = 1./cheby_calc_steps;
        }
    }
    else
    {
        cg_alphas.push_back(alpha);

        tea_leaf_cg_solve_calc_ur_device.setArg(0, alpha);

        //ENQUEUE(tea_leaf_cg_solve_calc_ur_device);
        ENQUEUE_OFFSET(tea_leaf_cg_solve_calc_ur_device);
        *rrn = reduceValue<double>(sum_red_kernels_double, reduce_buf_4);
    }
}

void CloverChunk::tea_leaf_kernel_cg_calc_p
(double beta)
{
    if (tea_solver == TEA_ENUM_CHEBYSHEV && cheby_calc_steps > 0)
    {
        return;
    }

    cg_betas.push_back(beta);

    tea_leaf_cg_solve_calc_p_device.setArg(0, beta);

    ENQUEUE(tea_leaf_cg_solve_calc_p_device);
    //ENQUEUE_OFFSET(tea_leaf_cg_solve_calc_p_device);

    //std::pair<double, double> eigs = calcEigenvalues(cg_alphas, cg_betas);
}

/********************/

// jacobi
void CloverChunk::tea_leaf_init_jacobi
(int coefficient, double dt, double * rx, double * ry)
{
    if (coefficient != CONDUCTIVITY && coefficient != RECIP_CONDUCTIVITY)
    {
        DIE("Unknown coefficient %d passed to tea leaf\n", coefficient);
    }

    calcrxry(dt, rx, ry);

    tea_leaf_jacobi_init_device.setArg(6, coefficient);
    //ENQUEUE(tea_leaf_jacobi_init_device);
    ENQUEUE_OFFSET(tea_leaf_jacobi_init_device);

    tea_leaf_jacobi_solve_device.setArg(0, *rx);
    tea_leaf_jacobi_solve_device.setArg(1, *ry);
}

void CloverChunk::tea_leaf_kernel_jacobi
(double rx, double ry, double* error)
{
    //ENQUEUE(tea_leaf_jacobi_copy_u_device);
    ENQUEUE_OFFSET(tea_leaf_jacobi_copy_u_device);
    //ENQUEUE(tea_leaf_jacobi_solve_device);
    ENQUEUE_OFFSET(tea_leaf_jacobi_solve_device);

    *error = reduceValue<double>(max_red_kernels_double, reduce_buf_1);
}

// both
void CloverChunk::tea_leaf_finalise
(void)
{
    //ENQUEUE(tea_leaf_finalise_device);
    ENQUEUE_OFFSET(tea_leaf_finalise_device);
}

