#include "ctx_common.hpp"

extern "C" void ocl_pack_buffers_
(int fields[NUM_FIELDS], int offsets[NUM_FIELDS], int * depth,
 int * face, double * host_buffer)
{
    tea_context.packUnpackAllBuffers(fields, offsets, *depth, *face, 1, host_buffer);
}

extern "C" void ocl_unpack_buffers_
(int fields[NUM_FIELDS], int offsets[NUM_FIELDS], int * depth,
 int * face, double * host_buffer)
{
    tea_context.packUnpackAllBuffers(fields, offsets, *depth, *face, 0, host_buffer);
}

void TeaCLContext::packUnpackAllBuffers
(int fields[NUM_FIELDS], int offsets[NUM_FIELDS],
 const int depth, const int face, const int pack,
 double * host_buffer)
{
    chunks.at(coarse_chunk)->packUnpackAllBuffers(fields, offsets, depth, face, pack, host_buffer);
}

