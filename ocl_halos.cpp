#include "ocl_common.hpp"
#include "kernel_files/ocl_kernel_hdr.hpp"
#include <sstream>
#include <iostream>

void CloverChunk::initHaloKernels
(void)
{
#ifndef CL_USE_HALO_KERNELS
    std::cout << "not using halo kernels" << std::endl;
#else
    std::stringstream ss;

    #define ADD_SOURCE(src_str) \
        ss << src_##src_str##_cl << std::endl;

    ADD_SOURCE(macros);
    ADD_SOURCE(halo_bare);

    #undef ADD_SOURCE

    #define COMPILE_ONE(dir, name) \
    { \
        try                                                         \
        {                                                           \
            cur_halo.dir = cl::Kernel(program, "update_halo_" #dir "_BARE");  \
            cur_halo.dir.setArg(1, name); \
        }                                                           \
        catch (cl::Error e){                                        \
            fprintf(stderr, "Error in creating %s kernel for %s\n",     \
                    #dir, #name);                                \
            exit(1);                                                \
        }                                                           \
    }

    // XXX messy
    const static cell_info_t CELL(    0, 0,  1,  1, 0, 0, CELL_DATA);
    const static cell_info_t VERTEX_X(1, 1, -1,  1, 0, 0, VERTEX_DATA);
    const static cell_info_t VERTEX_Y(1, 1,  1, -1, 0, 0, VERTEX_DATA);
    const static cell_info_t X_FACE(  1, 0, -1,  1, 1, 0, X_FACE_DATA);
    const static cell_info_t Y_FACE(  0, 1,  1, -1, 0, 1, Y_FACE_DATA);

    #define MAKE_HALO_KNLS(name, arr_type)   \
    {                                                               \
        halo_struct_t cur_halo; \
        std::stringstream options; \
        options << "-D LOCAL_X=" << LOCAL_X << " "; \
        options << "-D LOCAL_Y=" << LOCAL_Y << " "; \
        options << "-D CL_DEVICE_TYPE_GPU "; /* doesn't matter */ \
        options << "-D X_EXTRA=" << arr_type.x_extra << " "; \
        options << "-D Y_EXTRA=" << arr_type.y_extra << " "; \
        options << "-D X_INVERT=" << arr_type.x_invert << " "; \
        options << "-D Y_INVERT=" << arr_type.y_invert << " "; \
        options << "-D X_FACE=" << arr_type.x_face << " "; \
        options << "-D Y_FACE=" << arr_type.y_face << " "; \
        options << "-D GRID_TYPE=" << arr_type.grid_type << " "; \
        options << "-DCELL_DATA=" << CELL_DATA << " "; \
        options << "-DVERTEX_DATA=" << VERTEX_DATA << " "; \
        options << "-DX_FACE_DATA=" << X_FACE_DATA << " "; \
        options << "-DY_FACE_DATA=" << Y_FACE_DATA << " "; \
        options << "-Dx_min=" << x_min << " "; \
        options << "-Dx_max=" << x_max << " "; \
        options << "-Dy_min=" << y_min << " "; \
        options << "-Dy_max=" << y_max << " "; \
        fprintf(DBGOUT, "Making halo kernels for %s ", #name);    \
        compileProgram(ss.str(), options.str().c_str());                          \
        COMPILE_ONE(top, name); \
        COMPILE_ONE(bottom, name); \
        COMPILE_ONE(left, name); \
        COMPILE_ONE(right, name); \
        fprintf(DBGOUT, "Kernels for %s successfully built\n", #name); \
        cur_halo.wait_events = std::vector<cl::Event>(2); \
        halo_kernel_map[#name] = cur_halo; \
        fprintf(DBGOUT, "\n");                                      \
    }

    MAKE_HALO_KNLS(density0, CELL);
    MAKE_HALO_KNLS(density1, CELL);
    MAKE_HALO_KNLS(energy0, CELL);
    MAKE_HALO_KNLS(energy1, CELL);
    MAKE_HALO_KNLS(pressure, CELL);
    MAKE_HALO_KNLS(viscosity, CELL);

    MAKE_HALO_KNLS(xvel0, VERTEX_X);
    MAKE_HALO_KNLS(xvel1, VERTEX_X);

    MAKE_HALO_KNLS(yvel0, VERTEX_Y);
    MAKE_HALO_KNLS(yvel1, VERTEX_Y);

    MAKE_HALO_KNLS(vol_flux_x, X_FACE);
    MAKE_HALO_KNLS(mass_flux_x, X_FACE);

    MAKE_HALO_KNLS(vol_flux_y, Y_FACE);
    MAKE_HALO_KNLS(mass_flux_y, Y_FACE);
#endif
}
