#include "ocl_common.hpp"
extern CloverChunk chunk;

#define CHUNK_left          0
#define CHUNK_right         1
#define CHUNK_bottom        2
#define CHUNK_top           3

#define EXTERNAL_FACE       (-1)

// types of array data
const static cell_info_t CELL(    0, 0,  1,  1, 0, 0, CELL_DATA);
const static cell_info_t VERTEX_X(1, 1, -1,  1, 0, 0, VERTEX_DATA);
const static cell_info_t VERTEX_Y(1, 1,  1, -1, 0, 0, VERTEX_DATA);
const static cell_info_t X_FACE(  1, 0, -1,  1, 1, 0, X_FACE_DATA);
const static cell_info_t Y_FACE(  0, 1,  1, -1, 0, 1, Y_FACE_DATA);

extern "C" void update_halo_kernel_ocl_
(int *x_min,int *x_max,int *y_min,int *y_max,

const int* chunk_neighbours,

double* density0,
double* energy0,
double* pressure,
double* viscosity,
double* soundspeed,
double* density1,
double* energy1,
double* xvel0,
double* yvel0,
double* xvel1,
double* yvel1,
double* vol_flux_x,
double* vol_flux_y,
double* mass_flux_x,
double* mass_flux_y,
double* u,

const int* fields,
const int* depth)
{
    chunk.update_halo_kernel(fields, *depth, chunk_neighbours);
}

void CloverChunk::update_array
(cl::Buffer& cur_array,
const cell_info_t& array_type,
const int* chunk_neighbours,
int depth)
{
    // could do clenqueuecopybufferrect, but it's blocking and would be slow

    // could do offset launch for updating bottom/right, but dont to keep parity with cuda
    #define CHECK_LAUNCH(face, dir) \
    if(chunk_neighbours[CHUNK_ ## face] == EXTERNAL_FACE)\
    {\
        update_halo_##face##_device.setArg(0, array_type.x_extra); \
        update_halo_##face##_device.setArg(1, array_type.y_extra); \
        update_halo_##face##_device.setArg(2, array_type.x_invert); \
        update_halo_##face##_device.setArg(3, array_type.y_invert); \
        update_halo_##face##_device.setArg(4, array_type.x_face); \
        update_halo_##face##_device.setArg(5, array_type.y_face); \
        update_halo_##face##_device.setArg(6, array_type.grid_type); \
        update_halo_##face##_device.setArg(7, depth); \
        update_halo_##face##_device.setArg(8, cur_array); \
        CloverChunk::enqueueKernel(update_halo_##face##_device, \
                                   __LINE__, __FILE__,  \
                                   cl::NullRange,   \
                                   update_##dir##_global_size[depth-1], \
                                   update_##dir##_local_size[depth-1]); \
    }

    CHECK_LAUNCH(left, lr)
    CHECK_LAUNCH(right, lr)
    CHECK_LAUNCH(top, ud)
    CHECK_LAUNCH(bottom, ud)
}

void CloverChunk::update_halo_kernel
(const int* fields,
const int depth,
const int* chunk_neighbours)
{
    // if using CG, we want to update 'p' instead
    #define FIELD_work_array_1 FIELD_u

    #define HALO_UPDATE_RESIDENT(arr, type)                 \
    if(fields[FIELD_ ## arr] == 1)                          \
    {                                                       \
        update_array(arr, type, chunk_neighbours, depth);   \
    }

    HALO_UPDATE_RESIDENT(density0, CELL);
    HALO_UPDATE_RESIDENT(density1, CELL);
    HALO_UPDATE_RESIDENT(energy0, CELL);
    HALO_UPDATE_RESIDENT(energy1, CELL);
    HALO_UPDATE_RESIDENT(pressure, CELL);
    HALO_UPDATE_RESIDENT(viscosity, CELL);

    HALO_UPDATE_RESIDENT(xvel0, VERTEX_X);
    HALO_UPDATE_RESIDENT(xvel1, VERTEX_X);

    HALO_UPDATE_RESIDENT(yvel0, VERTEX_Y);
    HALO_UPDATE_RESIDENT(yvel1, VERTEX_Y);

    HALO_UPDATE_RESIDENT(vol_flux_x, X_FACE);
    HALO_UPDATE_RESIDENT(mass_flux_x, X_FACE);

    HALO_UPDATE_RESIDENT(vol_flux_y, Y_FACE);
    HALO_UPDATE_RESIDENT(mass_flux_y, Y_FACE);

    if (tl_use_cg)
    {
        // update 'p' instead
        HALO_UPDATE_RESIDENT(work_array_1, CELL);
    }
    else
    {
        HALO_UPDATE_RESIDENT(u, CELL);
    }
}

