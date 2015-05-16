#include "ocl_common.hpp"

// types of array data
const static cell_info_t CELL(    0, 0,  1,  1, 0, 0, CELL_DATA);

extern "C" void update_halo_kernel_ocl_
(const int* chunk_neighbours,
const int* fields,
const int* depth)
{
    tea_context.update_halo_kernel(fields, *depth, chunk_neighbours);
}

extern "C" void update_internal_halo_kernel_ocl_
(const int * chunk_neighbours,
 const int* fields,
 const int* depth)
{
    tea_context.update_internal_halo_kernel(fields, *depth, chunk_neighbours);
}

int TeaCLTile::isExternal
(int face) const
{
    return tile_external_faces[face - 1];
}

void TeaCLTile::setExternal
(int face)
{
    tile_external_faces[face - 1] = 1;
}

void TeaCLTile::update_array
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
        update_halo_##face##_device.setArg(0, array_type.x_extra); \
        update_halo_##face##_device.setArg(1, array_type.y_extra); \
        update_halo_##face##_device.setArg(2, array_type.x_invert); \
        update_halo_##face##_device.setArg(3, array_type.y_invert); \
        update_halo_##face##_device.setArg(4, array_type.x_face); \
        update_halo_##face##_device.setArg(5, array_type.y_face); \
        update_halo_##face##_device.setArg(6, array_type.grid_type); \
        update_halo_##face##_device.setArg(7, depth); \
        update_halo_##face##_device.setArg(8, cur_array); \
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

    #undef CHECK_LAUNCH
}

void TeaCLContext::update_halo_kernel
(const int* fields,
int depth,
const int* chunk_neighbours)
{
    #define HALO_UPDATE_RESIDENT(arr, type)                 \
    if(fields[FIELD_ ## arr - 1] == 1)                      \
    {                                                       \
        tile->update_array(tile->arr, type, chunk_neighbours, depth);   \
    }

    FOR_EACH_TILE
    {
        HALO_UPDATE_RESIDENT(density, CELL);
        HALO_UPDATE_RESIDENT(energy0, CELL);
        HALO_UPDATE_RESIDENT(energy1, CELL);
    }

    #undef HALO_UPDATE_RESIDENT
}

void TeaCLTile::packInternal
(cl::Buffer& cur_array,
 const cell_info_t& array_type,
 int depth)
{
    #define CHECK_PACK(face, dir) \
    if (tile_external_faces[CHUNK_ ## face - 1] == 0)   \
    {                                                           \
        pack_##face##_buffer_device.setArg(0, array_type.x_extra);    \
        pack_##face##_buffer_device.setArg(1, array_type.y_extra);    \
        pack_##face##_buffer_device.setArg(2, cur_array);         \
        pack_##face##_buffer_device.setArg(3, face##_buffer); \
        pack_##face##_buffer_device.setArg(4, depth);             \
        pack_##face##_buffer_device.setArg(5, 0);         \
        enqueueKernel(pack_##face##_buffer_device, \
                      __LINE__, __FILE__,  \
                      update_##dir##_offset[depth], \
                      update_##dir##_global_size[depth], \
                      update_##dir##_local_size[depth]); \
    }

    CHECK_PACK(bottom, bt)
    CHECK_PACK(top, bt)
    CHECK_PACK(left, lr)
    CHECK_PACK(right, lr)

    #undef CHECK_PACK
}

void TeaCLTile::unpackInternal
(cl::Buffer& cur_array,
 cl::Buffer& transferred_left,
 cl::Buffer& transferred_right,
 cl::Buffer& transferred_bottom,
 cl::Buffer& transferred_top,
 const cell_info_t& array_type,
 int depth)
{
    #define CHECK_UNPACK(face, dir)                       \
    if (tile_external_faces[CHUNK_ ## face - 1] == 0)   \
    {                                                           \
        unpack_##face##_buffer_device.setArg(0, array_type.x_extra);    \
        unpack_##face##_buffer_device.setArg(1, array_type.y_extra);    \
        unpack_##face##_buffer_device.setArg(2, cur_array);         \
        unpack_##face##_buffer_device.setArg(3, transferred_##face); \
        unpack_##face##_buffer_device.setArg(4, depth);             \
        unpack_##face##_buffer_device.setArg(5, 0);         \
        enqueueKernel(unpack_##face##_buffer_device, \
                      __LINE__, __FILE__,  \
                      update_##dir##_offset[depth], \
                      update_##dir##_global_size[depth], \
                      update_##dir##_local_size[depth]); \
    }

    CHECK_UNPACK(bottom, bt)
    CHECK_UNPACK(top, bt)
    CHECK_UNPACK(left, lr)
    CHECK_UNPACK(right, lr)

    #undef CHECK_UNPACK
}

void TeaCLContext::update_internal_halo_kernel
(const int* fields,
 int depth,
 const int* chunk_neighbours)
{
    #define HALO_UPDATE_PACK_INTERNAL(arr, type)                 \
    if(fields[FIELD_ ## arr - 1] == 1)                      \
    {                                                       \
        tile->packInternal(tile->arr, type, depth);   \
    }

    FOR_EACH_TILE
    {
        HALO_UPDATE_PACK_INTERNAL(density, CELL);
        HALO_UPDATE_PACK_INTERNAL(energy0, CELL);
        HALO_UPDATE_PACK_INTERNAL(energy1, CELL);
        HALO_UPDATE_PACK_INTERNAL(u, CELL);
        HALO_UPDATE_PACK_INTERNAL(vector_p, CELL);
        HALO_UPDATE_PACK_INTERNAL(vector_sd, CELL);
        HALO_UPDATE_PACK_INTERNAL(vector_r, CELL);
    }

    #undef HALO_UPDATE_INTERNAL

    // TODO exchange between tiles using clmigratememobjects

    #define HALO_UPDATE_UNPACK_INTERNAL(arr, type)                 \
    if(fields[FIELD_ ## arr - 1] == 1)                      \
    {                                                       \
        tile->unpackInternal(tile->arr, \
            tile->left_buffer, \
            tile->right_buffer, \
            tile->bottom_buffer, \
            tile->top_buffer, \
            type, depth);   \
    }

    FOR_EACH_TILE
    {
        HALO_UPDATE_UNPACK_INTERNAL(density, CELL);
        HALO_UPDATE_UNPACK_INTERNAL(energy0, CELL);
        HALO_UPDATE_UNPACK_INTERNAL(energy1, CELL);
        HALO_UPDATE_UNPACK_INTERNAL(u, CELL);
        HALO_UPDATE_UNPACK_INTERNAL(vector_p, CELL);
        HALO_UPDATE_UNPACK_INTERNAL(vector_sd, CELL);
        HALO_UPDATE_UNPACK_INTERNAL(vector_r, CELL);
    }

    #undef HALO_UPDATE_UNPACK_INTERNAL
}

