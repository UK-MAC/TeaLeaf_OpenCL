#include "ocl_common.hpp"

extern "C" void generate_chunk_kernel_ocl_
(const int* number_of_states,

const double* state_density,
const double* state_energy,
const double* state_xmin,
const double* state_xmax,
const double* state_ymin,
const double* state_ymax,
const double* state_radius,
const int* state_geometry,

const int* g_rect,
const int* g_circ,
const int* g_point)
{
    tea_context.generate_chunk_kernel(
        * number_of_states, state_density, state_energy,
        state_xmin, state_xmax, state_ymin, state_ymax,
        state_radius, state_geometry, * g_rect, * g_circ, *g_point);
}

void TeaCLContext::generate_chunk_kernel
(const int number_of_states, 
const double* state_density, const double* state_energy,
const double* state_xmin, const double* state_xmax,
const double* state_ymin, const double* state_ymax,
const double* state_radius, const int* state_geometry,
const int g_rect, const int g_circ, const int g_point)
{
    FOR_EACH_TILE
    {
        #define TEMP_ALLOC(arr)                                         \
            cl::Buffer tmp_state_##arr;                                 \
            try                                                         \
            {                                                           \
                tmp_state_##arr = cl::Buffer(context,                   \
                    CL_MEM_READ_ONLY,                                   \
                    number_of_states*sizeof(*state_##arr));             \
                tile->queue.enqueueWriteBuffer(tmp_state_##arr,               \
                    CL_TRUE,                                            \
                    0,                                                  \
                    number_of_states*sizeof(*state_##arr),              \
                    state_##arr);                                       \
            }                                                           \
            catch (cl::Error e)                                         \
            {                                                           \
                DIE("Error in creating %s buffer %d\n", #arr, e.err()); \
            }

        TEMP_ALLOC(density);
        TEMP_ALLOC(energy);
        TEMP_ALLOC(xmin);
        TEMP_ALLOC(xmax);
        TEMP_ALLOC(ymin);
        TEMP_ALLOC(ymax);
        TEMP_ALLOC(radius);
        TEMP_ALLOC(geometry);

        #undef TEMP_ALLOC

        tile->generate_chunk_init_device.setArg(2, tmp_state_density);
        tile->generate_chunk_init_device.setArg(3, tmp_state_energy);

        //ENQUEUE(generate_chunk_init_device);
        ENQUEUE_OFFSET(generate_chunk_init_device);

        tile->generate_chunk_device.setArg(6, tmp_state_density);
        tile->generate_chunk_device.setArg(7, tmp_state_energy);
        tile->generate_chunk_device.setArg(8, tmp_state_xmin);
        tile->generate_chunk_device.setArg(9, tmp_state_xmax);
        tile->generate_chunk_device.setArg(10, tmp_state_ymin);
        tile->generate_chunk_device.setArg(11, tmp_state_ymax);
        tile->generate_chunk_device.setArg(12, tmp_state_radius);
        tile->generate_chunk_device.setArg(13, tmp_state_geometry);

        tile->generate_chunk_device.setArg(14, g_rect);
        tile->generate_chunk_device.setArg(15, g_circ);
        tile->generate_chunk_device.setArg(16, g_point);
    }

    for (int state = 1; state < number_of_states; state++)
    {
        FOR_EACH_TILE
        {
            tile->generate_chunk_device.setArg(17, state);
        }

        //ENQUEUE(generate_chunk_device);
        ENQUEUE_OFFSET(generate_chunk_device);
    }

    FOR_EACH_TILE
    {
        tile->generate_chunk_init_u_device.setArg(1, tile->energy0);
    }
    ENQUEUE_OFFSET(generate_chunk_init_u_device);
}

