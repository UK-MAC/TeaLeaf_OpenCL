#include "ctx_common.hpp"

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
    chunks.at(fine_chunk)->generate_chunk_kernel(number_of_states,
        state_density, state_energy, state_xmin, state_xmax,
        state_ymin, state_ymax, state_radius, state_geometry,
        g_rect, g_circ, g_point);
}

