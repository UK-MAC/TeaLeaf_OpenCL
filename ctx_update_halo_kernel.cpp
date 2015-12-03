#include "ctx_common.hpp"

extern "C" void update_halo_kernel_ocl_
(const int* chunk_neighbours,
const int* fields,
const int* depth)
{
    tea_context.update_halo_kernel(fields, *depth, chunk_neighbours);
}

void TeaCLContext::update_halo_kernel
(const int* fields,
 int depth,
 const int* chunk_neighbours)
{
    chunks.at(fine_chunk)->update_halo_kernel(chunk_neighbours, fields, depth);
}

