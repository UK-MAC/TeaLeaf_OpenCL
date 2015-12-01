#include "ocl_common.hpp"

// types of array data
cell_info_t CELL = {.x_extra=0, .y_extra=0, .x_invert=1, .y_invert=1, .x_face=0, .y_face=0, .grid_type=CELL_DATA};

extern "C" void update_halo_kernel_ocl_
(const int* chunk_neighbours,
const int* fields,
const int* depth)
{
    tea_context.update_halo_kernel(fields, *depth, chunk_neighbours);
}

void TeaOpenCLTile::update_array
(cl::Buffer& cur_array,
const cell_info_t& array_type,
const int* chunk_neighbours,
int depth)
{
    // could do clenqueuecopybufferrect, but it's blocking and would be slow

    // could do offset launch for updating bottom/right, but dont to keep parity with cuda
    #define CHECK_LAUNCH(face, dir) \
    if(chunk_neighbours[CHUNK_ ## face - 1] == EXTERNAL_FACE)\
    {\
        update_halo_##face##_device.setArg(1, array_type); \
        update_halo_##face##_device.setArg(2, cur_array); \
        update_halo_##face##_device.setArg(3, depth); \
        enqueueKernel(update_halo_##face##_device, \
                      __LINE__, __FILE__,  \
                      update_##dir##_offset[depth], \
                      update_##dir##_global_size[depth], \
                      update_##dir##_local_size[depth]); \
    }

    CHECK_LAUNCH(bottom, bt)
    CHECK_LAUNCH(top, bt)
    CHECK_LAUNCH(left, lr)
    CHECK_LAUNCH(right, lr)
}

void TeaCLContext::update_halo_kernel
(const int* fields,
int depth,
const int* chunk_neighbours)
{
    #define HALO_UPDATE_RESIDENT(arr, type)                 \
    if (fields[FIELD_ ## arr - 1] == 1)                     \
    {                                                       \
        FOR_EACH_TILE                                       \
        {                                                   \
            tile->update_array(tile->arr, type, chunk_neighbours, depth);   \
        }                                                   \
    }

    HALO_UPDATE_RESIDENT(density, CELL);
    HALO_UPDATE_RESIDENT(energy0, CELL);
    HALO_UPDATE_RESIDENT(energy1, CELL);
}

