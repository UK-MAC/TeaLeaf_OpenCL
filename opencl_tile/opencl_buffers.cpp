#include "opencl_tile.hpp"

void TeaOpenCLChunk::initMemory
(void)
{
    if (!rank)
    {
        fprintf(stdout, "Allocating buffers\n");
    }

    size_t total_cells = (tile_x_cells+2*run_params.halo_exchange_depth+1) * (tile_y_cells+2*run_params.halo_exchange_depth+1);

    const std::vector<double> zeros(total_cells, 0.0);

    #define BUF_ALLOC(name, buf_sz)                 \
        try                                         \
        {                                           \
            name = cl::Buffer(context,              \
                              CL_MEM_READ_WRITE,    \
                              (buf_sz));            \
            queue.enqueueWriteBuffer(name,          \
                                     CL_TRUE,       \
                                     0,             \
                                     (buf_sz),      \
                                     &zeros[0]);    \
        }                                           \
        catch (cl::Error e)                         \
        {                                           \
            DIE("Error in creating %s buffer %d\n", \
                    #name, e.err());                \
        }

    #define BUF1DX_ALLOC(name, x_e)     \
        BUF_ALLOC(name, (tile_x_cells+2*run_params.halo_exchange_depth+x_e) * sizeof(double))

    #define BUF1DY_ALLOC(name, y_e)     \
        BUF_ALLOC(name, (tile_y_cells+2*run_params.halo_exchange_depth+y_e) * sizeof(double))

    #define BUF2D_ALLOC(name, x_e, y_e) \
        BUF_ALLOC(name, (tile_x_cells+2*run_params.halo_exchange_depth+x_e) * (tile_y_cells+2*run_params.halo_exchange_depth+y_e) * sizeof(double))

    BUF2D_ALLOC(density, 0, 0);
    BUF2D_ALLOC(energy0, 0, 0);
    BUF2D_ALLOC(energy1, 0, 0);

    BUF2D_ALLOC(volume, 0, 0);
    BUF2D_ALLOC(xarea, 1, 0);
    BUF2D_ALLOC(yarea, 0, 1);

    BUF1DX_ALLOC(cellx, 0);
    BUF1DX_ALLOC(celldx, 0);
    BUF1DX_ALLOC(vertexx, 1);
    BUF1DX_ALLOC(vertexdx, 1);

    BUF1DY_ALLOC(celly, 0);
    BUF1DY_ALLOC(celldy, 0);
    BUF1DY_ALLOC(vertexy, 1);
    BUF1DY_ALLOC(vertexdy, 1);

    // work arrays used in various kernels (post_vol, pre_vol, mom_flux, etc)
    BUF2D_ALLOC(vector_p, 1, 1);
    BUF2D_ALLOC(vector_r, 1, 1);
    BUF2D_ALLOC(vector_w, 1, 1);
    BUF2D_ALLOC(vector_Mi, 1, 1);
    BUF2D_ALLOC(vector_Kx, 1, 1);
    BUF2D_ALLOC(vector_Ky, 1, 1);
    BUF2D_ALLOC(vector_sd, 1, 1);

    // tealeaf
    BUF2D_ALLOC(u, 0, 0);
    BUF2D_ALLOC(u0, 0, 0);
    BUF2D_ALLOC(vector_z, 1, 1);

    BUF2D_ALLOC(vector_rm1, 1, 1);

    BUF2D_ALLOC(cp, 0, 0);
    BUF2D_ALLOC(bfp, 0, 0);

    // When these are allocated, we don't need any kind of padding round the edges
    #define BUF2D_ALLOC_SMALL(name, x_e, y_e) \
        BUF_ALLOC(name, local_coarse_x_cells*local_coarse_y_cells*sizeof(double))

    BUF2D_ALLOC_SMALL(coarse_local_Kx, 0, 0);
    BUF2D_ALLOC_SMALL(coarse_local_Ky, 0, 0);
    BUF2D_ALLOC_SMALL(coarse_local_t2, 0, 0);
    BUF2D_ALLOC_SMALL(coarse_local_ztr, 0, 0);
    BUF2D_ALLOC_SMALL(coarse_local_ztaz, 0, 0);

    // allocate enough for 1 item per work group, and then a bit extra for the reduction
    // 1.5 should work even if wg size is 2
    size_t reduce_buf_sz = 1.5*((sizeof(double)*reduced_cells)/(LOCAL_X*LOCAL_Y));
    BUF_ALLOC(reduce_buf_1, reduce_buf_sz);
    BUF_ALLOC(reduce_buf_2, reduce_buf_sz);
    BUF_ALLOC(reduce_buf_3, reduce_buf_sz);
    BUF_ALLOC(reduce_buf_4, reduce_buf_sz);
    BUF_ALLOC(reduce_buf_5, reduce_buf_sz);
    BUF_ALLOC(reduce_buf_6, reduce_buf_sz);

    queue.enqueueWriteBuffer(reduce_buf_1, CL_TRUE, 0, reduce_buf_sz, &zeros.front());
    queue.enqueueWriteBuffer(reduce_buf_2, CL_TRUE, 0, reduce_buf_sz, &zeros.front());
    queue.enqueueWriteBuffer(reduce_buf_3, CL_TRUE, 0, reduce_buf_sz, &zeros.front());
    queue.enqueueWriteBuffer(reduce_buf_4, CL_TRUE, 0, reduce_buf_sz, &zeros.front());
    queue.enqueueWriteBuffer(reduce_buf_5, CL_TRUE, 0, reduce_buf_sz, &zeros.front());
    queue.enqueueWriteBuffer(reduce_buf_6, CL_TRUE, 0, reduce_buf_sz, &zeros.front());

    // size of one side of mesh, plus one extra on the side for each depth, times the number of halos to be exchanged
    size_t lr_mpi_buf_sz = sizeof(double)*(tile_y_cells + 2*run_params.halo_exchange_depth)*run_params.halo_exchange_depth;
    size_t bt_mpi_buf_sz = sizeof(double)*(tile_x_cells + 2*run_params.halo_exchange_depth)*run_params.halo_exchange_depth;

    // enough for 1 for each array - overkill, but not that much extra space
    BUF_ALLOC(left_buffer, NUM_BUFFERED_FIELDS*lr_mpi_buf_sz);
    BUF_ALLOC(right_buffer, NUM_BUFFERED_FIELDS*lr_mpi_buf_sz);
    BUF_ALLOC(bottom_buffer, NUM_BUFFERED_FIELDS*bt_mpi_buf_sz);
    BUF_ALLOC(top_buffer, NUM_BUFFERED_FIELDS*bt_mpi_buf_sz);

    #undef BUF2D_ALLOC
    #undef BUF2D_ALLOC_SMALL
    #undef BUF1DX_ALLOC
    #undef BUF1DY_ALLOC
    #undef BUF_ALLOC

 #define ADD_BUFFER_DBG_MAP(name) arr_names[#name] = name;
    ADD_BUFFER_DBG_MAP(u);
    ADD_BUFFER_DBG_MAP(u0);
    ADD_BUFFER_DBG_MAP(cp);
    ADD_BUFFER_DBG_MAP(bfp);

    ADD_BUFFER_DBG_MAP(vector_sd);
    ADD_BUFFER_DBG_MAP(vector_p);
    ADD_BUFFER_DBG_MAP(vector_r);
    ADD_BUFFER_DBG_MAP(vector_w);
    ADD_BUFFER_DBG_MAP(vector_Mi);
    ADD_BUFFER_DBG_MAP(vector_Kx);
    ADD_BUFFER_DBG_MAP(vector_Ky);

    ADD_BUFFER_DBG_MAP(density);
    ADD_BUFFER_DBG_MAP(energy0);
    ADD_BUFFER_DBG_MAP(energy1);

    ADD_BUFFER_DBG_MAP(cellx);
    ADD_BUFFER_DBG_MAP(celly);
    ADD_BUFFER_DBG_MAP(celldx);
    ADD_BUFFER_DBG_MAP(celldy);
    ADD_BUFFER_DBG_MAP(vertexx);
    ADD_BUFFER_DBG_MAP(vertexy);
    ADD_BUFFER_DBG_MAP(vertexdx);
    ADD_BUFFER_DBG_MAP(vertexdy);
#undef ADD_BUFFER_DBG_MAP

    MPI_Barrier(MPI_COMM_WORLD);

    if (!rank)
    {
        fprintf(stdout, "Buffers allocated\n");
    }
}

