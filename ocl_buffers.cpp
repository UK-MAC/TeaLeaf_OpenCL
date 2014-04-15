#include "ocl_common.hpp"

void CloverChunk::initBuffers
(void)
{
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
        BUF_ALLOC(name, (x_max+4+x_e) * sizeof(double))

    #define BUF1DY_ALLOC(name, y_e)     \
        BUF_ALLOC(name, (y_max+4+y_e) * sizeof(double))

    #define BUF2D_ALLOC(name, x_e, y_e) \
        BUF_ALLOC(name, (x_max+4+x_e) * (y_max+4+y_e) * sizeof(double))

    BUF2D_ALLOC(density0, 0, 0);
    BUF2D_ALLOC(density1, 0, 0);
    BUF2D_ALLOC(energy0, 0, 0);
    BUF2D_ALLOC(energy1, 0, 0);

    BUF2D_ALLOC(pressure, 0, 0);
    BUF2D_ALLOC(soundspeed, 0, 0);
    BUF2D_ALLOC(viscosity, 0, 0);
    BUF2D_ALLOC(volume, 0, 0);

    BUF2D_ALLOC(xvel0, 1, 1);
    BUF2D_ALLOC(xvel1, 1, 1);
    BUF2D_ALLOC(yvel0, 1, 1);
    BUF2D_ALLOC(yvel1, 1, 1);

    BUF2D_ALLOC(xarea, 1, 0);
    BUF2D_ALLOC(vol_flux_x, 1, 0);
    BUF2D_ALLOC(mass_flux_x, 1, 0);

    BUF2D_ALLOC(yarea, 0, 1);
    BUF2D_ALLOC(vol_flux_y, 0, 1);
    BUF2D_ALLOC(mass_flux_y, 0, 1);

    BUF1DX_ALLOC(cellx, 0);
    BUF1DX_ALLOC(celldx, 0);
    BUF1DX_ALLOC(vertexx, 1);
    BUF1DX_ALLOC(vertexdx, 1);

    BUF1DY_ALLOC(celly, 0);
    BUF1DY_ALLOC(celldy, 0);
    BUF1DY_ALLOC(vertexy, 1);
    BUF1DY_ALLOC(vertexdy, 1);

    // work arrays used in various kernels (post_vol, pre_vol, mom_flux, etc)
    BUF2D_ALLOC(work_array_1, 1, 1);
    BUF2D_ALLOC(work_array_2, 1, 1);
    BUF2D_ALLOC(work_array_3, 1, 1);
    BUF2D_ALLOC(work_array_4, 1, 1);
    BUF2D_ALLOC(work_array_5, 1, 1);

    // tealeaf
    BUF2D_ALLOC(u, 0, 0);
    BUF2D_ALLOC(z, 1, 1);
    BUF2D_ALLOC(work_array_6, 1, 1);

    // allocate enough for 1 item per work group, and then a bit extra for the reduction
    // 1.5 should work even if wg size is 2
    BUF_ALLOC(reduce_buf_1, 1.5*((sizeof(double)*reduced_cells)/(LOCAL_X*LOCAL_Y)));
    BUF_ALLOC(reduce_buf_2, 1.5*((sizeof(double)*reduced_cells)/(LOCAL_X*LOCAL_Y)));
    BUF_ALLOC(reduce_buf_3, 1.5*((sizeof(double)*reduced_cells)/(LOCAL_X*LOCAL_Y)));
    BUF_ALLOC(reduce_buf_4, 1.5*((sizeof(double)*reduced_cells)/(LOCAL_X*LOCAL_Y)));
    BUF_ALLOC(reduce_buf_5, 1.5*((sizeof(double)*reduced_cells)/(LOCAL_X*LOCAL_Y)));
    BUF_ALLOC(reduce_buf_6, 1.5*((sizeof(double)*reduced_cells)/(LOCAL_X*LOCAL_Y)));
    BUF_ALLOC(PdV_reduce_buf, 1.5*((sizeof(int)*reduced_cells)/(LOCAL_X*LOCAL_Y)));

    #undef BUF2D_ALLOC
    #undef BUF1DX_ALLOC
    #undef BUF1DY_ALLOC
    #undef BUF_ALLOC

    fprintf(DBGOUT, "Buffers allocated\n");

 #define ADD_BUFFER_DBG_MAP(name) arr_names[#name] = name;
    ADD_BUFFER_DBG_MAP(volume);
    ADD_BUFFER_DBG_MAP(soundspeed);
    ADD_BUFFER_DBG_MAP(pressure);
    ADD_BUFFER_DBG_MAP(viscosity);

    ADD_BUFFER_DBG_MAP(u);

    ADD_BUFFER_DBG_MAP(density0);
    ADD_BUFFER_DBG_MAP(density1);
    ADD_BUFFER_DBG_MAP(energy0);
    ADD_BUFFER_DBG_MAP(energy1);
    ADD_BUFFER_DBG_MAP(xvel0);
    ADD_BUFFER_DBG_MAP(xvel1);
    ADD_BUFFER_DBG_MAP(yvel0);
    ADD_BUFFER_DBG_MAP(yvel1);
    ADD_BUFFER_DBG_MAP(xarea);
    ADD_BUFFER_DBG_MAP(yarea);
    ADD_BUFFER_DBG_MAP(vol_flux_x);
    ADD_BUFFER_DBG_MAP(vol_flux_y);
    ADD_BUFFER_DBG_MAP(mass_flux_x);
    ADD_BUFFER_DBG_MAP(mass_flux_y);

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

