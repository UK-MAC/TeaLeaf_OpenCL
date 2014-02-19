#include "ocl_common.hpp"
extern CloverChunk chunk;

extern "C" void advec_cell_kernel_ocl_
(const int* xmin, const int* xmax, const int* ymin, const int* ymax,
const int* dr,
const int* swp_nmbr,
const bool* vector,
const double* vertexdx,
const double* vertexdy,
const double* volume,
double* density1,
double* energy1,
double* mass_flux_x,
const double* vol_flux_x,
double* mass_flux_y,
const double* vol_flux_y,

double* unused_array1,
double* unused_array2,
double* unused_array3,
double* unused_array4,
double* unused_array5,
double* unused_array6,
double* unused_array7)
{
    chunk.advec_cell_kernel(*dr, *swp_nmbr);
}

void CloverChunk::advec_cell_kernel
(int dr, int swp_nmbr)
{
    if (1 == dr)
    {
        advec_cell_pre_vol_x_device.setArg(0, swp_nmbr);
        advec_cell_ener_flux_x_device.setArg(0, swp_nmbr);
        advec_cell_x_device.setArg(0, swp_nmbr);

        ENQUEUE(advec_cell_pre_vol_x_device);
        ENQUEUE(advec_cell_ener_flux_x_device);
        ENQUEUE(advec_cell_x_device);
    }
    else
    {
        advec_cell_pre_vol_y_device.setArg(0, swp_nmbr);
        advec_cell_ener_flux_y_device.setArg(0, swp_nmbr);
        advec_cell_y_device.setArg(0, swp_nmbr);

        ENQUEUE(advec_cell_pre_vol_y_device);
        ENQUEUE(advec_cell_ener_flux_y_device);
        ENQUEUE(advec_cell_y_device);
    }

    // XXX testing
    //tea_leaf_kernel();
}

