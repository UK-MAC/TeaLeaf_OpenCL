#include "ocl_common.hpp"

void CloverChunk::initBuffers
(void)
{
    size_t total_cells = (x_max+2*halo_exchange_depth+1) * (y_max+2*halo_exchange_depth+1);
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
        BUF_ALLOC(name, (x_max+2*halo_exchange_depth+x_e) * sizeof(double))

    #define BUF1DY_ALLOC(name, y_e)     \
        BUF_ALLOC(name, (y_max+2*halo_exchange_depth+y_e) * sizeof(double))

    #define BUF2D_ALLOC(name, x_e, y_e) \
        BUF_ALLOC(name, (x_max+2*halo_exchange_depth+x_e) * (y_max+2*halo_exchange_depth+y_e) * sizeof(double))

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

    BUF2D_ALLOC(cp, 0, 0);
    BUF2D_ALLOC(bfp, 0, 0);

    // allocate enough for 1 item per work group, and then a bit extra for the reduction
    // 1.5 should work even if wg size is 2
    BUF_ALLOC(reduce_buf_1, 1.5*((sizeof(double)*reduced_cells)/(LOCAL_X*LOCAL_Y)));
    BUF_ALLOC(reduce_buf_2, 1.5*((sizeof(double)*reduced_cells)/(LOCAL_X*LOCAL_Y)));
    BUF_ALLOC(reduce_buf_3, 1.5*((sizeof(double)*reduced_cells)/(LOCAL_X*LOCAL_Y)));
    BUF_ALLOC(reduce_buf_4, 1.5*((sizeof(double)*reduced_cells)/(LOCAL_X*LOCAL_Y)));
    BUF_ALLOC(reduce_buf_5, 1.5*((sizeof(double)*reduced_cells)/(LOCAL_X*LOCAL_Y)));
    BUF_ALLOC(reduce_buf_6, 1.5*((sizeof(double)*reduced_cells)/(LOCAL_X*LOCAL_Y)));

    // size of one side of mesh, plus one extra on the side for each depth, times the number of halos to be exchanged
    size_t lr_mpi_buf_sz = sizeof(double)*(y_max + 2*halo_exchange_depth)*halo_exchange_depth;
    size_t bt_mpi_buf_sz = sizeof(double)*(x_max + 2*halo_exchange_depth)*halo_exchange_depth;

    // enough for 1 for each array - overkill, but not that much extra space
    BUF_ALLOC(left_buffer, NUM_BUFFERED_FIELDS*lr_mpi_buf_sz);
    BUF_ALLOC(right_buffer, NUM_BUFFERED_FIELDS*lr_mpi_buf_sz);
    BUF_ALLOC(bottom_buffer, NUM_BUFFERED_FIELDS*bt_mpi_buf_sz);
    BUF_ALLOC(top_buffer, NUM_BUFFERED_FIELDS*bt_mpi_buf_sz);

    fprintf(DBGOUT, "Buffers allocated\n");

    #undef BUF2D_ALLOC
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
}

