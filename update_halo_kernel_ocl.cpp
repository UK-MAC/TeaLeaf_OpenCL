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
#ifdef CL_USE_HALO_KERNELS
    #define top_EVENT_IDX 0
    #define bottom_EVENT_IDX 1

    #define CHECK_LAUNCH_HORZ(arr, face, dir)                                               \
    if(chunk_neighbours[CHUNK_ ## face] == EXTERNAL_FACE)                                   \
    {                                                                                       \
        halo_kernel_map[#arr].face.setArg(0, depth);                                        \
        CloverChunk::enqueueKernel(halo_kernel_map[#arr].face,                              \
                                   __LINE__, __FILE__,                                      \
                                   cl::NullRange,                                           \
                                   update_##dir##_global_size[depth-1],                     \
                                   update_##dir##_local_size[depth-1],                      \
                                   NULL,                                                    \
                                   &halo_kernel_map[#arr].wait_events[face##_EVENT_IDX]);   \
    }                                                                                       \
    else                                                                                    \
    {                                                                                       \
       halo_kernel_map[#arr].wait_events[face##_EVENT_IDX] = cl::Event();                   \
    }

    #define CHECK_LAUNCH_VERT(arr, face, dir)                           \
    if(chunk_neighbours[CHUNK_ ## face] == EXTERNAL_FACE)               \
    {                                                                   \
        halo_kernel_map[#arr].face.setArg(0, depth);                    \
        CloverChunk::enqueueKernel(halo_kernel_map[#arr].face,          \
                                   __LINE__, __FILE__,                  \
                                   cl::NullRange,                       \
                                   update_##dir##_global_size[depth-1], \
                                   update_##dir##_local_size[depth-1],  \
                                   &halo_kernel_map[#arr].wait_events,  \
                                   NULL);                               \
    }

    #define HALO_UPDATE_2(arr)                          \
    if(fields[FIELD_ ## arr] == 1)                      \
    {                                                   \
        CHECK_LAUNCH_HORZ(arr, top, ud);                \
        CHECK_LAUNCH_HORZ(arr, bottom, ud);             \
        CHECK_LAUNCH_VERT(arr, left, lr);               \
        CHECK_LAUNCH_VERT(arr, right, lr);              \
    }

    HALO_UPDATE_2(density0);
    HALO_UPDATE_2(density1);
    HALO_UPDATE_2(energy0);
    HALO_UPDATE_2(energy1);
    HALO_UPDATE_2(pressure);
    HALO_UPDATE_2(viscosity);

    HALO_UPDATE_2(xvel0);
    HALO_UPDATE_2(xvel1);

    HALO_UPDATE_2(yvel0);
    HALO_UPDATE_2(yvel1);

    HALO_UPDATE_2(vol_flux_x);
    HALO_UPDATE_2(mass_flux_x);

    HALO_UPDATE_2(vol_flux_y);
    HALO_UPDATE_2(mass_flux_y);

#else

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

    HALO_UPDATE_RESIDENT(u, CELL);
#endif
}

