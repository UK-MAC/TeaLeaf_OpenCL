#include "ocl_common.hpp"
#include "ocl_kernel_hdr.hpp"
#include <sstream>

void CloverChunk::initProgram
(void)
{
    // options
    std::stringstream options("");

    // FIXME check to make sure its nvidia
    if (desired_type == CL_DEVICE_TYPE_GPU)
    {
        // for nvidia architecture
        //options << "-cl-nv-arch " << NV_ARCH << " ";
        //options << "-cl-nv-maxrregcount=20 ";
    }

#ifdef __arm__
    // on ARM, don't use built in functions as they don't exist
    options << "-DCLOVER_NO_BUILTINS ";
#endif

#if defined(NO_KERNEL_REDUCTIONS)
    // don't do any reductions inside the kernels
    options << "-D NO_KERNEL_REDUCTIONS ";
#endif

#ifdef ONED_KERNEL_LAUNCHES
    // launch kernels with 1d work group size
    options << "-DONED_KERNEL_LAUNCHES ";
#endif

    // use jacobi preconditioner when running CG solver
    options << "-DCG_DO_PRECONDITION ";

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

    // include current directory
    options << "-I. ";

    // device type in the form "-D..."
    options << device_type_prepro;

    const std::string options_str = options.str();

    fprintf(DBGOUT, "Compiling kernels with options:\n%s\n", options_str.c_str());
    fprintf(stdout, "Compiling kernels (may take some time)...");
    fflush(stdout);

    compileKernel(options_str, src_ideal_gas_cl, "ideal_gas", ideal_gas_device);
    compileKernel(options_str, src_accelerate_cl, "accelerate", accelerate_device);
    compileKernel(options_str, src_flux_calc_cl, "flux_calc_x", flux_calc_x_device);
    compileKernel(options_str, src_flux_calc_cl, "flux_calc_y", flux_calc_y_device);
    compileKernel(options_str, src_viscosity_cl, "viscosity", viscosity_device);
    compileKernel(options_str, src_revert_cl, "revert", revert_device);

    compileKernel(options_str, src_initialise_chunk_cl, "initialise_chunk_first", initialise_chunk_first_device);
    compileKernel(options_str, src_initialise_chunk_cl, "initialise_chunk_second", initialise_chunk_second_device);
    compileKernel(options_str, src_generate_chunk_cl, "generate_chunk_init", generate_chunk_init_device);
    compileKernel(options_str, src_generate_chunk_cl, "generate_chunk", generate_chunk_device);

    compileKernel(options_str, src_reset_field_cl, "reset_field", reset_field_device);
    compileKernel(options_str, src_set_field_cl, "set_field", set_field_device);

    compileKernel(options_str, src_PdV_cl, "PdV_predict", PdV_predict_device);
    compileKernel(options_str, src_PdV_cl, "PdV_not_predict", PdV_not_predict_device);

    compileKernel(options_str, src_field_summary_cl, "field_summary", field_summary_device);
    compileKernel(options_str, src_calc_dt_cl, "calc_dt", calc_dt_device);

    compileKernel(options_str, src_update_halo_cl, "update_halo_top", update_halo_top_device);
    compileKernel(options_str, src_update_halo_cl, "update_halo_bottom", update_halo_bottom_device);
    compileKernel(options_str, src_update_halo_cl, "update_halo_left", update_halo_left_device);
    compileKernel(options_str, src_update_halo_cl, "update_halo_right", update_halo_right_device);

    compileKernel(options_str, src_advec_mom_cl, "advec_mom_vol", advec_mom_vol_device);
    compileKernel(options_str, src_advec_mom_cl, "advec_mom_node_flux_post_x_1", advec_mom_node_flux_post_x_1_device);
    compileKernel(options_str, src_advec_mom_cl, "advec_mom_node_flux_post_x_2", advec_mom_node_flux_post_x_2_device);
    compileKernel(options_str, src_advec_mom_cl, "advec_mom_node_pre_x", advec_mom_node_pre_x_device);
    compileKernel(options_str, src_advec_mom_cl, "advec_mom_flux_x", advec_mom_flux_x_device);
    compileKernel(options_str, src_advec_mom_cl, "advec_mom_xvel", advec_mom_xvel_device);
    compileKernel(options_str, src_advec_mom_cl, "advec_mom_node_flux_post_y_1", advec_mom_node_flux_post_y_1_device);
    compileKernel(options_str, src_advec_mom_cl, "advec_mom_node_flux_post_y_2", advec_mom_node_flux_post_y_2_device);
    compileKernel(options_str, src_advec_mom_cl, "advec_mom_node_pre_y", advec_mom_node_pre_y_device);
    compileKernel(options_str, src_advec_mom_cl, "advec_mom_flux_y", advec_mom_flux_y_device);
    compileKernel(options_str, src_advec_mom_cl, "advec_mom_yvel", advec_mom_yvel_device);

    compileKernel(options_str, src_advec_cell_cl, "advec_cell_pre_vol_x", advec_cell_pre_vol_x_device);
    compileKernel(options_str, src_advec_cell_cl, "advec_cell_ener_flux_x", advec_cell_ener_flux_x_device);
    compileKernel(options_str, src_advec_cell_cl, "advec_cell_x", advec_cell_x_device);
    compileKernel(options_str, src_advec_cell_cl, "advec_cell_pre_vol_y", advec_cell_pre_vol_y_device);
    compileKernel(options_str, src_advec_cell_cl, "advec_cell_ener_flux_y", advec_cell_ener_flux_y_device);
    compileKernel(options_str, src_advec_cell_cl, "advec_cell_y", advec_cell_y_device);

    if (tea_solver == TEA_ENUM_CG)
    {
        compileKernel(options_str, src_tea_leaf_cg_cl, "tea_leaf_cg_init_u", tea_leaf_cg_init_u_device);
        compileKernel(options_str, src_tea_leaf_cg_cl, "tea_leaf_cg_init_directions", tea_leaf_cg_init_directions_device);
        compileKernel(options_str, src_tea_leaf_cg_cl, "tea_leaf_cg_init_others", tea_leaf_cg_init_others_device);
        compileKernel(options_str, src_tea_leaf_cg_cl, "tea_leaf_cg_solve_calc_w", tea_leaf_cg_solve_calc_w_device);
        compileKernel(options_str, src_tea_leaf_cg_cl, "tea_leaf_cg_solve_calc_ur", tea_leaf_cg_solve_calc_ur_device);
        compileKernel(options_str, src_tea_leaf_cg_cl, "tea_leaf_cg_solve_calc_p", tea_leaf_cg_solve_calc_p_device);
    }
    else
    {
        compileKernel(options_str, src_tea_leaf_jacobi_cl, "tea_leaf_jacobi_init", tea_leaf_jacobi_init_device);
        compileKernel(options_str, src_tea_leaf_jacobi_cl, "tea_leaf_jacobi_copy_u", tea_leaf_jacobi_copy_u_device);
        compileKernel(options_str, src_tea_leaf_jacobi_cl, "tea_leaf_jacobi_solve", tea_leaf_jacobi_solve_device);
    }

    compileKernel(options_str, src_tea_leaf_jacobi_cl, "tea_leaf_finalise", tea_leaf_finalise_device);

    fprintf(stdout, "done.\n");
    fprintf(DBGOUT, "All kernels compiled\n");
}

void CloverChunk::compileKernel
(const std::string& options,
 const std::string& source_name,
 const char* kernel_name,
 cl::Kernel& kernel)
{
    const std::string source_str(source_name);
    fprintf(DBGOUT, "Compiling %s...", kernel_name);
    cl::Program program;

    if (built_programs.find(source_name) == built_programs.end())
    {
        try
        {
            program = compileProgram(source_str, options);
        }
        catch (KernelCompileError err)
        {
            DIE("Errors in compiling %s:\n%s\n", kernel_name, err.what());
        }

        built_programs[source_name] = program;
    }
    else
    {
        program = built_programs.at(source_name);
    }

    size_t max_wg_size;

    try
    {
        kernel = cl::Kernel(program, kernel_name);
    }
    catch (cl::Error e)
    {
        fprintf(DBGOUT, "Failed\n");
        DIE("Error in creating %s kernel %d\n",
                kernel_name, e.err());
    }
    cl::detail::errHandler(
        clGetKernelWorkGroupInfo(kernel(),
                                 device(),
                                 CL_KERNEL_WORK_GROUP_SIZE,
                                 sizeof(size_t),
                                 &max_wg_size,
                                 NULL));
    if ((LOCAL_X*LOCAL_Y) > max_wg_size)
    {
        DIE("Work group size %zux%zu is too big for kernel %s"
            " - maximum is %zu\n",
                LOCAL_X, LOCAL_Y, kernel_name,
                max_wg_size);
    }

    fprintf(DBGOUT, "Done\n");
    fflush(DBGOUT);
}

cl::Program CloverChunk::compileProgram
(const std::string& source,
 const std::string& options)
{
    // catches any warnings/errors in the build
    std::stringstream errstream("");

    // very verbose
    //fprintf(stderr, "Making with source:\n%s\n", source.c_str());
    //fprintf(DBGOUT, "Making with options string:\n%s\n", options.c_str());
    fflush(DBGOUT);
    cl::Program program;

    cl::Program::Sources sources;
    sources = cl::Program::Sources(1, std::make_pair(source.c_str(), source.length()));

    try
    {
        program = cl::Program(context, sources);
        std::vector<cl::Device> dev_vec(1, device);
        program.build(dev_vec, options.c_str());
    }
    catch (cl::Error e)
    {
        fprintf(stderr, "Errors in creating program\n");

        try
        {
            errstream << program.getBuildInfo<CL_PROGRAM_BUILD_LOG>(device);
        }
        catch (cl::Error ie)
        {
            DIE("Error in retrieving build info\n");
        }

        std::string errs(errstream.str());
        //DIE("%s\n", errs.c_str());
        throw KernelCompileError(errs.c_str());
    }

    // return
    errstream << program.getBuildInfo<CL_PROGRAM_BUILD_LOG>(device);
    std::string errs(errstream.str());

    // some will print out an empty warning log
    if (errs.size() > 10)
    {
        fprintf(DBGOUT, "Warnings:\n%s\n", errs.c_str());
    }

    return program;
}

void CloverChunk::initSizes
(void)
{
#if defined(ONED_KERNEL_LAUNCHES)
    size_t glob_x = x_max+5;
    size_t glob_y = y_max+5;
    total_cells = glob_x*glob_y;

    // pad as below
    while (total_cells % LOCAL_X)
    {
        total_cells++;
    }

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

    if (CL_DEVICE_TYPE_ACCELERATOR == desired_type)
    {
        // on xeon phi, needs to be 16 so that update left/right kernels dont go really slow
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
    #define FIND_PADDING_SIZE(knl, vmin, vmax, hmin, hmax)                      \
    {                                                                            \
        size_t global_horz_size = (-(hmin)) + (hmax) + x_max;                   \
        while (global_horz_size % LOCAL_X) global_horz_size++;                  \
        size_t global_vert_size = (-(vmin)) + (vmax) + y_max;                   \
        while (global_vert_size % LOCAL_Y) global_vert_size++;                  \
        launch_specs_t cur_specs;                                               \
        cur_specs.global = cl::NDRange(global_horz_size, global_vert_size);     \
        cur_specs.offset = cl::NDRange(x_min + 1 + (vmin), y_min + 1 + (hmin)); \
        launch_specs[#knl"_device"] = cur_specs;                                \
    }

    FIND_PADDING_SIZE(ideal_gas, 0, 0, 0, 0); // works
    FIND_PADDING_SIZE(accelerate, 0, 1, 0, 1); // works
    FIND_PADDING_SIZE(flux_calc_x, 0, 0, 0, 1); // works
    FIND_PADDING_SIZE(flux_calc_y, 0, 0, 1, 0); // works
    FIND_PADDING_SIZE(viscosity, 0, 0, 0, 0); // works
    FIND_PADDING_SIZE(revert, 0, 0, 0, 0); // works
    FIND_PADDING_SIZE(reset_field, 0, 1, 0, 1); // works
    FIND_PADDING_SIZE(set_field, 0, 1, 0, 1); // works
    FIND_PADDING_SIZE(field_summary, 0, 0, 0, 0);
    FIND_PADDING_SIZE(calc_dt, 0, 0, 0, 0);

    FIND_PADDING_SIZE(advec_mom_vol, -2, 2, -2, 2); // works
    FIND_PADDING_SIZE(advec_mom_node_flux_post_x_1, 0, 1, -2, 2);
    FIND_PADDING_SIZE(advec_mom_node_flux_post_x_2, 0, 1, -1, 2);
    FIND_PADDING_SIZE(advec_mom_node_pre_x, 0, 1, -1, 2); // works
    FIND_PADDING_SIZE(advec_mom_flux_x, 0, 1, -1, 1); // works
    FIND_PADDING_SIZE(advec_mom_xvel, 0, 1, 0, 1); // works
    FIND_PADDING_SIZE(advec_mom_node_flux_post_y_1, -2, 2, 0, 1);
    FIND_PADDING_SIZE(advec_mom_node_flux_post_y_2, -1, 2, 0, 1);
    FIND_PADDING_SIZE(advec_mom_node_pre_y, -1, 2, 0, 1); // works
    FIND_PADDING_SIZE(advec_mom_flux_y, -1, 1, 0, 1); // works
    FIND_PADDING_SIZE(advec_mom_yvel, 0, 1, 0, 1); // works

    FIND_PADDING_SIZE(advec_cell_pre_vol_x, -2, 2, -2, 2); // works
    FIND_PADDING_SIZE(advec_cell_ener_flux_x, 0, 0, 0, 2); // works
    FIND_PADDING_SIZE(advec_cell_x, 0, 0, 0, 0); // works
    FIND_PADDING_SIZE(advec_cell_pre_vol_y, -2, 2, -2, 2); // works
    FIND_PADDING_SIZE(advec_cell_ener_flux_y, 0, 2, 0, 2); // works
    FIND_PADDING_SIZE(advec_cell_y, 0, 0, 0, 0); // works

    FIND_PADDING_SIZE(PdV_predict, 0, 0, 0, 0); // works
    FIND_PADDING_SIZE(PdV_not_predict, 0, 0, 0, 0); // works

    FIND_PADDING_SIZE(initialise_chunk_first, 0, 3, 0, 3);
    FIND_PADDING_SIZE(initialise_chunk_second, -2, 2, -2, 2);
    FIND_PADDING_SIZE(generate_chunk_init, -2, 2, -2, 2);
    FIND_PADDING_SIZE(generate_chunk, -2, 2, -2, 2);

    FIND_PADDING_SIZE(generate_chunk, -2, 2, -2, 2);

    if (tea_solver == TEA_ENUM_CG)
    {
        FIND_PADDING_SIZE(tea_leaf_cg_init_u, -2, 2, -2, 2); // works
        FIND_PADDING_SIZE(tea_leaf_cg_init_directions, 0, 1, 0, 1); // works
        FIND_PADDING_SIZE(tea_leaf_cg_init_others, 0, 0, 0, 0);
        FIND_PADDING_SIZE(tea_leaf_cg_solve_calc_w, 0, 0, 0, 0);
        FIND_PADDING_SIZE(tea_leaf_cg_solve_calc_ur, 0, 0, 0, 0);
        FIND_PADDING_SIZE(tea_leaf_cg_solve_calc_p, 0, 0, 0, 0); // works
    }
    else
    {
        FIND_PADDING_SIZE(tea_leaf_jacobi_init, -1, 1, -1, 1);
        FIND_PADDING_SIZE(tea_leaf_jacobi_copy_u, -1, 1, -1, 1);
        FIND_PADDING_SIZE(tea_leaf_jacobi_solve, 0, 0, 0, 0);
    }

    FIND_PADDING_SIZE(tea_leaf_jacobi_finalise, 0, 0, 0, 0);
}

void CloverChunk::initArgs
(void)
{
    #define SETARG_CHECK(knl, idx, buf) \
        try \
        { \
            knl.setArg(idx, buf); \
        } \
        catch (cl::Error e) \
        { \
            DIE("Error in setting argument index %d to %s for kernel %s (%s - %d)", \
                idx, #buf, #knl, \
                e.what(), e.err()); \
        }

    // ideal_gas
    ideal_gas_device.setArg(2, pressure);
    ideal_gas_device.setArg(3, soundspeed);

    // accelerate
    accelerate_device.setArg(1, xarea);
    accelerate_device.setArg(2, yarea);
    accelerate_device.setArg(3, volume);
    accelerate_device.setArg(4, density0);
    accelerate_device.setArg(5, pressure);
    accelerate_device.setArg(6, viscosity);
    accelerate_device.setArg(7, xvel0);
    accelerate_device.setArg(8, yvel0);
    accelerate_device.setArg(9, xvel1);
    accelerate_device.setArg(10, yvel1);

    // flux calc
    flux_calc_x_device.setArg(1, xarea);
    flux_calc_x_device.setArg(2, xvel0);
    flux_calc_x_device.setArg(3, xvel1);
    flux_calc_x_device.setArg(4, vol_flux_x);

    flux_calc_y_device.setArg(1, yarea);
    flux_calc_y_device.setArg(2, yvel0);
    flux_calc_y_device.setArg(3, yvel1);
    flux_calc_y_device.setArg(4, vol_flux_y);

    // viscosity
    viscosity_device.setArg(0, celldx);
    viscosity_device.setArg(1, celldy);
    viscosity_device.setArg(2, density0);
    viscosity_device.setArg(3, pressure);
    viscosity_device.setArg(4, viscosity);
    viscosity_device.setArg(5, xvel0);
    viscosity_device.setArg(6, yvel0);

    // revert
    revert_device.setArg(0, density0);
    revert_device.setArg(1, density1);
    revert_device.setArg(2, energy0);
    revert_device.setArg(3, energy1);

    // initialise chunk
    initialise_chunk_first_device.setArg(4, vertexx);
    initialise_chunk_first_device.setArg(5, vertexdx);
    initialise_chunk_first_device.setArg(6, vertexy);
    initialise_chunk_first_device.setArg(7, vertexdy);

    initialise_chunk_second_device.setArg(4, vertexx);
    initialise_chunk_second_device.setArg(5, vertexdx);
    initialise_chunk_second_device.setArg(6, vertexy);
    initialise_chunk_second_device.setArg(7, vertexdy);
    initialise_chunk_second_device.setArg(8, cellx);
    initialise_chunk_second_device.setArg(9, celldx);
    initialise_chunk_second_device.setArg(10, celly);
    initialise_chunk_second_device.setArg(11, celldy);
    initialise_chunk_second_device.setArg(12, volume);
    initialise_chunk_second_device.setArg(13, xarea);
    initialise_chunk_second_device.setArg(14, yarea);

    // advec_mom
    /*
    post_vol = work array 1
    node_flux = work array 2 _AND_ pre_vol = work array 2
    node_mass_post = work array 3
    node_mass_pre = work array 4
    mom_flux = work array 5
    */
    advec_mom_vol_device.setArg(1, work_array_1);
    advec_mom_vol_device.setArg(2, work_array_2);
    advec_mom_vol_device.setArg(3, volume);
    advec_mom_vol_device.setArg(4, vol_flux_x);
    advec_mom_vol_device.setArg(5, vol_flux_y);

    // x kernels
    advec_mom_node_flux_post_x_1_device.setArg(0, work_array_2);
    advec_mom_node_flux_post_x_1_device.setArg(1, mass_flux_x);

    advec_mom_node_flux_post_x_2_device.setArg(0, work_array_3);
    advec_mom_node_flux_post_x_2_device.setArg(1, work_array_1);
    advec_mom_node_flux_post_x_2_device.setArg(2, density1);

    advec_mom_node_pre_x_device.setArg(0, work_array_2);
    advec_mom_node_pre_x_device.setArg(1, work_array_3);
    advec_mom_node_pre_x_device.setArg(2, work_array_4);

    advec_mom_flux_x_device.setArg(0, work_array_2);
    advec_mom_flux_x_device.setArg(1, work_array_3);
    advec_mom_flux_x_device.setArg(2, work_array_4);
    advec_mom_flux_x_device.setArg(4, celldx);
    advec_mom_flux_x_device.setArg(5, work_array_5);

    advec_mom_xvel_device.setArg(0, work_array_3);
    advec_mom_xvel_device.setArg(1, work_array_4);
    advec_mom_xvel_device.setArg(2, work_array_5);

    // y kernels
    advec_mom_node_flux_post_y_1_device.setArg(0, work_array_2);
    advec_mom_node_flux_post_y_1_device.setArg(1, mass_flux_y);

    advec_mom_node_flux_post_y_2_device.setArg(0, work_array_3);
    advec_mom_node_flux_post_y_2_device.setArg(1, work_array_1);
    advec_mom_node_flux_post_y_2_device.setArg(2, density1);

    advec_mom_node_pre_y_device.setArg(0, work_array_2);
    advec_mom_node_pre_y_device.setArg(1, work_array_3);
    advec_mom_node_pre_y_device.setArg(2, work_array_4);

    advec_mom_flux_y_device.setArg(0, work_array_2);
    advec_mom_flux_y_device.setArg(1, work_array_3);
    advec_mom_flux_y_device.setArg(2, work_array_4);
    advec_mom_flux_y_device.setArg(4, celldy);
    advec_mom_flux_y_device.setArg(5, work_array_5);

    advec_mom_yvel_device.setArg(0, work_array_3);
    advec_mom_yvel_device.setArg(1, work_array_4);
    advec_mom_yvel_device.setArg(2, work_array_5);

    // advec cell
    /*
    post_vol = work array 1 _AND_ ener_flux = work_array_1
    pre_vol = work array 2
    */

    #define SET_SHARED(knl)             \
        knl.setArg(1, volume);          \
        knl.setArg(2, vol_flux_x);      \
        knl.setArg(3, vol_flux_y);      \
        knl.setArg(4, work_array_2);    \
        knl.setArg(5, density1);        \
        knl.setArg(6, energy1);         \
        knl.setArg(7, work_array_1);

    // x kernels
    advec_cell_pre_vol_x_device.setArg(1, work_array_2);
    advec_cell_pre_vol_x_device.setArg(2, work_array_1);
    advec_cell_pre_vol_x_device.setArg(3, volume);
    advec_cell_pre_vol_x_device.setArg(4, vol_flux_x);
    advec_cell_pre_vol_x_device.setArg(5, vol_flux_y);

    SET_SHARED(advec_cell_ener_flux_x_device)
    advec_cell_ener_flux_x_device.setArg(8, vertexdx);
    advec_cell_ener_flux_x_device.setArg(9, mass_flux_x);

    SET_SHARED(advec_cell_x_device)
    advec_cell_x_device.setArg(8, mass_flux_x);

    // y kernels
    advec_cell_pre_vol_y_device.setArg(1, work_array_2);
    advec_cell_pre_vol_y_device.setArg(2, work_array_1);
    advec_cell_pre_vol_y_device.setArg(3, volume);
    advec_cell_pre_vol_y_device.setArg(4, vol_flux_x);
    advec_cell_pre_vol_y_device.setArg(5, vol_flux_y);

    SET_SHARED(advec_cell_ener_flux_y_device)
    advec_cell_ener_flux_y_device.setArg(8, vertexdy);
    advec_cell_ener_flux_y_device.setArg(9, mass_flux_y);

    SET_SHARED(advec_cell_y_device)
    advec_cell_y_device.setArg(8, mass_flux_y);

    #undef SET_SHARED

    // reset field
    reset_field_device.setArg(0, density0);
    reset_field_device.setArg(1, density1);
    reset_field_device.setArg(2, energy0);
    reset_field_device.setArg(3, energy1);
    reset_field_device.setArg(4, xvel0);
    reset_field_device.setArg(5, xvel1);
    reset_field_device.setArg(6, yvel0);
    reset_field_device.setArg(7, yvel1);

    // set field
    set_field_device.setArg(0, density0);
    set_field_device.setArg(1, density1);
    set_field_device.setArg(2, energy0);
    set_field_device.setArg(3, energy1);

    // generate chunk
    generate_chunk_init_device.setArg(0, density0);
    generate_chunk_init_device.setArg(1, energy0);
    generate_chunk_init_device.setArg(2, xvel0);
    generate_chunk_init_device.setArg(3, yvel0);

    generate_chunk_device.setArg(0, vertexx);
    generate_chunk_device.setArg(1, vertexy);
    generate_chunk_device.setArg(2, cellx);
    generate_chunk_device.setArg(3, celly);
    generate_chunk_device.setArg(4, density0);
    generate_chunk_device.setArg(5, energy0);
    generate_chunk_device.setArg(6, xvel0);
    generate_chunk_device.setArg(7, yvel0);
    generate_chunk_device.setArg(8, u);

    // PdV
    PdV_predict_device.setArg(1, PdV_reduce_buf);
    PdV_predict_device.setArg(2, xarea);
    PdV_predict_device.setArg(3, yarea);
    PdV_predict_device.setArg(4, volume);
    PdV_predict_device.setArg(5, density0);
    PdV_predict_device.setArg(6, density1);
    PdV_predict_device.setArg(7, energy0);
    PdV_predict_device.setArg(8, energy1);
    PdV_predict_device.setArg(9, pressure);
    PdV_predict_device.setArg(10, viscosity);
    PdV_predict_device.setArg(11, xvel0);
    PdV_predict_device.setArg(12, yvel0);
    PdV_predict_device.setArg(13, xvel1);
    PdV_predict_device.setArg(14, yvel1);

    PdV_not_predict_device.setArg(1, PdV_reduce_buf);
    PdV_not_predict_device.setArg(2, xarea);
    PdV_not_predict_device.setArg(3, yarea);
    PdV_not_predict_device.setArg(4, volume);
    PdV_not_predict_device.setArg(5, density0);
    PdV_not_predict_device.setArg(6, density1);
    PdV_not_predict_device.setArg(7, energy0);
    PdV_not_predict_device.setArg(8, energy1);
    PdV_not_predict_device.setArg(9, pressure);
    PdV_not_predict_device.setArg(10, viscosity);
    PdV_not_predict_device.setArg(11, xvel0);
    PdV_not_predict_device.setArg(12, yvel0);
    PdV_not_predict_device.setArg(13, xvel1);
    PdV_not_predict_device.setArg(14, yvel1);

    // field summary
    field_summary_device.setArg(0, volume);
    field_summary_device.setArg(1, density0);
    field_summary_device.setArg(2, energy0);
    field_summary_device.setArg(3, pressure);
    field_summary_device.setArg(4, xvel0);
    field_summary_device.setArg(5, yvel0);
    field_summary_device.setArg(6, u);

    field_summary_device.setArg(7, reduce_buf_1);
    field_summary_device.setArg(8, reduce_buf_2);
    field_summary_device.setArg(9, reduce_buf_3);
    field_summary_device.setArg(10, reduce_buf_4);
    field_summary_device.setArg(11, reduce_buf_5);
    field_summary_device.setArg(12, reduce_buf_6);

    // calc dt
    /*
    work_array_1 = jk_ctrl
    work_array_2 = dt_min
    */
    calc_dt_device.setArg(7, xarea);
    calc_dt_device.setArg(8, yarea);
    calc_dt_device.setArg(9, celldx);
    calc_dt_device.setArg(10, celldy);
    calc_dt_device.setArg(11, volume);
    calc_dt_device.setArg(12, density0);
    calc_dt_device.setArg(13, viscosity);
    calc_dt_device.setArg(14, soundspeed);
    calc_dt_device.setArg(15, xvel0);
    calc_dt_device.setArg(16, xvel0);
    calc_dt_device.setArg(17, reduce_buf_1);
    calc_dt_device.setArg(18, reduce_buf_2);

    // no parameters set for update_halo here

    // tealeaf
    if (tea_solver == TEA_ENUM_CG)
    {
        /*
         *  work_array_1 = p
         *  work_array_2 = r
         *  work_array_3 = w / d (just for initialisation)
         *  work_array_4 = Mi
         *
         *  work_array_5 = Kx
         *  work_array_6 = Ky
         *
         *  reduce_buf_1 = bb
         *  reduce_buf_2 = rro
         *  reduce_buf_3 = pw
         *  reduce_buf_4 = rrn
         */
        tea_leaf_cg_init_u_device.setArg(0, density1);
        tea_leaf_cg_init_u_device.setArg(1, energy1);
        tea_leaf_cg_init_u_device.setArg(2, u);
        tea_leaf_cg_init_u_device.setArg(3, work_array_1);
        tea_leaf_cg_init_u_device.setArg(4, work_array_2);
        tea_leaf_cg_init_u_device.setArg(5, work_array_3);

        tea_leaf_cg_init_directions_device.setArg(0, work_array_3);
        tea_leaf_cg_init_directions_device.setArg(1, work_array_5);
        tea_leaf_cg_init_directions_device.setArg(2, work_array_6);

        tea_leaf_cg_init_others_device.setArg(0, reduce_buf_2);
        tea_leaf_cg_init_others_device.setArg(1, u);
        tea_leaf_cg_init_others_device.setArg(2, work_array_1);
        tea_leaf_cg_init_others_device.setArg(3, work_array_2);
        tea_leaf_cg_init_others_device.setArg(4, work_array_3);
        tea_leaf_cg_init_others_device.setArg(5, work_array_4);
        tea_leaf_cg_init_others_device.setArg(6, work_array_5);
        tea_leaf_cg_init_others_device.setArg(7, work_array_6);
        // used when preconditioner is used
        tea_leaf_cg_init_others_device.setArg(10, z);

        tea_leaf_cg_solve_calc_w_device.setArg(0, reduce_buf_3);
        tea_leaf_cg_solve_calc_w_device.setArg(1, work_array_1);
        tea_leaf_cg_solve_calc_w_device.setArg(2, work_array_3);
        tea_leaf_cg_solve_calc_w_device.setArg(3, work_array_5);
        tea_leaf_cg_solve_calc_w_device.setArg(4, work_array_6);

        tea_leaf_cg_solve_calc_ur_device.setArg(1, reduce_buf_4);
        tea_leaf_cg_solve_calc_ur_device.setArg(2, u);
        tea_leaf_cg_solve_calc_ur_device.setArg(3, work_array_1);
        tea_leaf_cg_solve_calc_ur_device.setArg(4, work_array_2);
        tea_leaf_cg_solve_calc_ur_device.setArg(5, work_array_3);
        // used when preconditioner is used
        tea_leaf_cg_solve_calc_ur_device.setArg(6, z);
        tea_leaf_cg_solve_calc_ur_device.setArg(7, work_array_4);

        tea_leaf_cg_solve_calc_p_device.setArg(1, work_array_1);
        tea_leaf_cg_solve_calc_p_device.setArg(2, work_array_2);
        tea_leaf_cg_solve_calc_p_device.setArg(3, z);
    }
    else
    {
        tea_leaf_jacobi_init_device.setArg(0, density1);
        tea_leaf_jacobi_init_device.setArg(1, energy1);
        tea_leaf_jacobi_init_device.setArg(2, work_array_1);
        tea_leaf_jacobi_init_device.setArg(3, work_array_2);
        tea_leaf_jacobi_init_device.setArg(4, work_array_3);
        tea_leaf_jacobi_init_device.setArg(5, u);

        tea_leaf_jacobi_copy_u_device.setArg(0, u);
        tea_leaf_jacobi_copy_u_device.setArg(1, work_array_4);

        tea_leaf_jacobi_solve_device.setArg(2, work_array_1);
        tea_leaf_jacobi_solve_device.setArg(3, work_array_2);
        tea_leaf_jacobi_solve_device.setArg(4, work_array_3);
        tea_leaf_jacobi_solve_device.setArg(5, u);
        tea_leaf_jacobi_solve_device.setArg(6, work_array_4);
        tea_leaf_jacobi_solve_device.setArg(7, reduce_buf_1);
    }

    // both finalise the same
    tea_leaf_finalise_device.setArg(0, density1);
    tea_leaf_finalise_device.setArg(1, u);
    tea_leaf_finalise_device.setArg(2, energy1);

    fprintf(DBGOUT, "Kernel arguments set\n");
}

