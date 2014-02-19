#include "ocl_common.hpp"
#include "kernel_files/ocl_kernel_hdr.hpp"

#include <sstream>

void CloverChunk::initProgram
(void)
{
    // add all the sources into one big string
    std::stringstream ss;
    #define ADD_SOURCE(src_str) \
        ss << src_##src_str##_cl << std::endl;

    // add macros first
    ADD_SOURCE(macros);

    // then add kernels
    ADD_SOURCE(ideal_gas);
    ADD_SOURCE(accelerate);
    ADD_SOURCE(flux_calc);
    ADD_SOURCE(viscosity);
    ADD_SOURCE(revert);
    ADD_SOURCE(initialise_chunk);
    ADD_SOURCE(advec_mom);
    ADD_SOURCE(advec_cell);
    ADD_SOURCE(reset_field);
    ADD_SOURCE(generate_chunk);
    ADD_SOURCE(PdV);
    ADD_SOURCE(field_summary);
    ADD_SOURCE(calc_dt);
	// has to be included last! FIXME
    ADD_SOURCE(update_halo);

    #undef ADD_SOURCE

    // options
    std::stringstream options("");

    // FIXME check to make sure its nvidia
    if (desired_type == CL_DEVICE_TYPE_GPU)
    {
        // for nvidia architecture
        //options << "-cl-nv-arch " << NV_ARCH << " ";
        //options << "-cl-nv-maxrregcount=20 ";
    }

    // on ARM, don't use built in functions as they don't exist
#ifdef __arm__
    options << "-DCLOVER_NO_BUILTINS ";
    options << "-w ";
#endif

    // pass in these values so you don't have to pass them in to every kernel
    options << "-Dx_min=" << x_min << " ";
    options << "-Dx_max=" << x_max << " ";
    options << "-Dy_min=" << y_min << " ";
    options << "-Dy_max=" << y_max << " ";

    // local sizes
    options << "-DBLOCK_SZ=" << LOCAL_X*LOCAL_Y << " ";
    options << "-DLOCAL_X=" << LOCAL_X << " ";
    options << "-DLOCAL_Y=" << LOCAL_Y << " ";

    // for update halo
    options << "-DCELL_DATA=" << CELL_DATA << " ";
    options << "-DVERTEX_DATA=" << VERTEX_DATA << " ";
    options << "-DX_FACE_DATA=" << X_FACE_DATA << " ";
    options << "-DY_FACE_DATA=" << Y_FACE_DATA << " ";

#ifdef ONED_KERNEL_LAUNCHES
    options << "-DONED_KERNEL_LAUNCHES ";
#endif

    // not in 1.2 - not needed anyway
    //options << "-cl-strict-aliasing ";

    // device type in the form "-D..."
    options << device_type_prepro;

    fprintf(DBGOUT, "Compiling kernels with options:\n%s\n", options.str().c_str());

    compileProgram(ss.str(), options.str());

    size_t max_wg_size;
    #define COMPILE_KERNEL(kernel_name)                                 \
        try                                                             \
        {                                                               \
            fprintf(DBGOUT, "Compiling %s\n", #kernel_name); \
            kernel_name##_device = cl::Kernel(program, #kernel_name);   \
        }                                                               \
        catch (cl::Error e){                                            \
            fprintf(stderr, "Error in creating %s kernel %d\n",         \
                    #kernel_name, e.err());                             \
            exit(1);                                                    \
        }                                                               \
        cl::detail::errHandler(                                         \
            clGetKernelWorkGroupInfo(kernel_name##_device(),            \
                                     device(),                          \
                                     CL_KERNEL_WORK_GROUP_SIZE,         \
                                     sizeof(size_t),                    \
                                     &max_wg_size,                      \
                                     NULL));                            \
        if ((LOCAL_X*LOCAL_Y) > max_wg_size)                            \
        {                                                               \
            fprintf(stderr, "Work group size %zux%zu is too big for kernel %s", \
                    LOCAL_X, LOCAL_Y, #kernel_name);                    \
            fprintf(stderr, " - maximum is %zu\n", max_wg_size);        \
            exit(1); \
        }

    COMPILE_KERNEL(ideal_gas);
    COMPILE_KERNEL(accelerate);
    COMPILE_KERNEL(flux_calc);
    COMPILE_KERNEL(viscosity);
    COMPILE_KERNEL(revert);
    COMPILE_KERNEL(reset_field);
    COMPILE_KERNEL(field_summary);
    COMPILE_KERNEL(calc_dt);

    // initialise chunk kernels
    COMPILE_KERNEL(initialise_chunk_first);
    COMPILE_KERNEL(initialise_chunk_second);

    // generate chunk kernels
    COMPILE_KERNEL(generate_chunk_init)
    COMPILE_KERNEL(generate_chunk)

    // various advec_mom kernels
    COMPILE_KERNEL(advec_mom_vol)
    COMPILE_KERNEL(advec_mom_node_flux_post_x)
    COMPILE_KERNEL(advec_mom_node_pre_x)
    COMPILE_KERNEL(advec_mom_flux_x)
    COMPILE_KERNEL(advec_mom_xvel)
    COMPILE_KERNEL(advec_mom_node_flux_post_y)
    COMPILE_KERNEL(advec_mom_node_pre_y)
    COMPILE_KERNEL(advec_mom_flux_y)
    COMPILE_KERNEL(advec_mom_yvel)

    // various advec_cell kernels
    COMPILE_KERNEL(advec_cell_pre_vol_x)
    COMPILE_KERNEL(advec_cell_ener_flux_x)
    COMPILE_KERNEL(advec_cell_x)
    COMPILE_KERNEL(advec_cell_pre_vol_y)
    COMPILE_KERNEL(advec_cell_ener_flux_y)
    COMPILE_KERNEL(advec_cell_y)

    // PdV kernels
    COMPILE_KERNEL(PdV_predict)
    COMPILE_KERNEL(PdV_not_predict)

    // update halo
    COMPILE_KERNEL(update_halo_bottom)
    COMPILE_KERNEL(update_halo_top)
    COMPILE_KERNEL(update_halo_right)
    COMPILE_KERNEL(update_halo_left)

    #undef COMPILE_KERNEL

    fprintf(DBGOUT, "All kernels compiled\n");
}

void CloverChunk::initSizes
(void)
{
#ifdef ONED_KERNEL_LAUNCHES
    size_t glob_x = x_max+5;
    size_t glob_y = y_max+5;
    total_cells = glob_x*glob_y;

    // pad as below
    while (total_cells % LOCAL_X)
    {
        total_cells++;
    }

    fprintf(DBGOUT, "Local size = %zu\n", LOCAL_X);
    fprintf(DBGOUT, "Global size = %zu\n", total_cells);
    global_size = cl::NDRange(total_cells);
#else
    fprintf(DBGOUT, "Local size = %zux%zu\n", LOCAL_X, LOCAL_Y);

    // pad the global size so the local size fits
    size_t glob_x = x_max+5 +
        (((x_max+5)%LOCAL_X == 0) ? 0 : (LOCAL_X - ((x_max+5)%LOCAL_X)));
    size_t glob_y = y_max+5 +
        (((y_max+5)%LOCAL_Y == 0) ? 0 : (LOCAL_Y - ((y_max+5)%LOCAL_Y)));
    total_cells = glob_x*glob_y;

    fprintf(DBGOUT, "Global size = %zux%zu\n", glob_x, glob_y);
    global_size = cl::NDRange(glob_x, glob_y);
#endif

    /*
     *  update halo kernels need specific work group sizes - not doing a
     *  reduction, so can just fit it to the row/column even if its not a pwoer
     *  of 2
     */
    // get max local size for the update kernels
    size_t max_update_wg_sz;
    cl::detail::errHandler(
        clGetKernelWorkGroupInfo(update_halo_bottom_device(),
                                 device(),
                                 CL_KERNEL_WORK_GROUP_SIZE,
                                 sizeof(size_t),
                                 &max_update_wg_sz,
                                 NULL));
    fprintf(DBGOUT, "Max work group size for update halo is %zu\n", max_update_wg_sz);

    // subdivide row size until it will fit
    size_t local_row_size = x_max+5;
    while (local_row_size > max_update_wg_sz/2)
    {
        local_row_size = local_row_size/2;
    }

    // xeon phi does not like large numbers of weird work groups
    if (CL_DEVICE_TYPE_ACCELERATOR == desired_type)
    {
        local_row_size = 16;
    }

    fprintf(DBGOUT, "Local row work group size is %zu\n", local_row_size);

    update_ud_local_size[0] = cl::NDRange(local_row_size, 1);
    update_ud_local_size[1] = cl::NDRange(local_row_size, 2);

    size_t global_row_size = local_row_size;
    while (global_row_size < x_max+5)
    {
        global_row_size += local_row_size;
    }
    update_ud_global_size[0] = cl::NDRange(global_row_size, 1);
    update_ud_global_size[1] = cl::NDRange(global_row_size, 2);

    // same for column
    size_t local_column_size = y_max+5;
    while (local_column_size > max_update_wg_sz/2)
    {
        local_column_size = local_column_size/2;
    }

    // as above - xeon phi
    if (CL_DEVICE_TYPE_ACCELERATOR == desired_type)
    {
        local_column_size = 16;
    }

    fprintf(DBGOUT, "Local column work group size is %zu\n", local_column_size);

    update_lr_local_size[0] = cl::NDRange(1, local_column_size);
    update_lr_local_size[1] = cl::NDRange(2, local_column_size);

    size_t global_column_size = local_column_size;
    while (global_column_size < y_max+5)
    {
        global_column_size += local_column_size;
    }
    update_lr_global_size[0] = cl::NDRange(1, global_column_size);
    update_lr_global_size[1] = cl::NDRange(2, global_column_size);

    fprintf(DBGOUT, "Update halo parameters calculated\n");

    /*
     *  figure out offset launch sizes for the various kernels
     *  no 'smart' way to do this?
     */
    #define FIND_PADDING_SIZE(knl, rmin, rmax, cmin, cmax)                      \
    {                                                                           \
        size_t global_row_size = (-(rmin)) + (rmax) + x_max;                    \
        while (global_row_size % LOCAL_X) global_row_size++;                    \
        size_t global_col_size = (-(cmin)) + (cmax) + y_max;                    \
        while (global_col_size % LOCAL_Y) global_col_size++;                    \
        launch_specs_t cur_specs;                                               \
        cur_specs.global = cl::NDRange(global_row_size, global_col_size);       \
        cur_specs.offset = cl::NDRange(x_min + 1 + (rmin), y_min + 1 + (cmin)); \
        launch_specs[#knl"_device"] = cur_specs;                                \
        /*fprintf(stdout, "Kernel:%s\n global:%zu %zu\n offset:%zu %zu\n",      \
            #knl,                                                               \
            launch_specs.at(#knl "_device").global[0],                          \
            launch_specs.at(#knl "_device").global[1],                          \
            launch_specs.at(#knl "_device").offset[0],                          \
            launch_specs.at(#knl "_device").offset[1]);*/                       \
    }

    FIND_PADDING_SIZE(ideal_gas, 0, 0, 0, 0); // works
    FIND_PADDING_SIZE(accelerate, 0, 1, 0, 1); // works
    FIND_PADDING_SIZE(flux_calc, 0, 0, 1, 1); // works
    FIND_PADDING_SIZE(viscosity, 0, 0, 0, 0); // works
    FIND_PADDING_SIZE(revert, 0, 0, 0, 0); // works
    FIND_PADDING_SIZE(reset_field, 0, 1, 0, 1); // works
    FIND_PADDING_SIZE(field_summary, 0, 0, 0, 0);
    FIND_PADDING_SIZE(calc_dt, 0, 0, 0, 0);
    FIND_PADDING_SIZE(advec_mom_vol, -2, 2, -2, 2);
    //FIND_PADDING_SIZE(advec_mom_node_flux_post_x);
    FIND_PADDING_SIZE(advec_mom_node_pre_x, 0, 1, -1, 2);
    FIND_PADDING_SIZE(advec_mom_flux_x, 0, 1, -1, 1);
    FIND_PADDING_SIZE(advec_mom_xvel, 0, 1, 0, 1);
    //FIND_PADDING_SIZE(advec_mom_node_flux_post_y);
    FIND_PADDING_SIZE(advec_mom_node_pre_y, -1, 2, 0, 1);
    FIND_PADDING_SIZE(advec_mom_flux_y, -1, 1, 0, 1);
    FIND_PADDING_SIZE(advec_mom_yvel, 0, 1, 0, 1);
    FIND_PADDING_SIZE(advec_cell_pre_vol_x, -2, 2, -2, 2);
    FIND_PADDING_SIZE(advec_cell_ener_flux_x, 0, 0, 0, 2);
    FIND_PADDING_SIZE(advec_cell_x, 0, 0, 0, 0);
    FIND_PADDING_SIZE(advec_cell_pre_vol_y, -2, 2, -2, 2);
    FIND_PADDING_SIZE(advec_cell_ener_flux_y, 0, 2, 0, 0);
    FIND_PADDING_SIZE(advec_cell_y, 0, 0, 0, 0);
    FIND_PADDING_SIZE(PdV_predict, 0, 0, 0, 0); // works
    FIND_PADDING_SIZE(PdV_not_predict, 0, 0, 0, 0); // works

    FIND_PADDING_SIZE(initialise_chunk_first, 0, 3, 0, 3);
    FIND_PADDING_SIZE(initialise_chunk_second, -2, 2, -2, 2);
    FIND_PADDING_SIZE(generate_chunk_init, -2, 2, -2, 2);
    FIND_PADDING_SIZE(generate_chunk, -2, 2, -2, 2);
}

