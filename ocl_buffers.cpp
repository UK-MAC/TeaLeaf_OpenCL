#include "ocl_common.hpp"

void TeaCLContext::initMemory
(void)
{
    if (!rank)
    {
        fprintf(stdout, "Allocating buffers\n");
    }

    FOR_EACH_TILE
    {
        tile_it->second->initMemory();
    }

    MPI_Barrier(MPI_COMM_WORLD);
    if (!rank)
    {
        fprintf(stdout, "Buffers allocated\n");
    }
}

