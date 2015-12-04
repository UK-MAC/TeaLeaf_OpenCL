
MODULE tea_leaf_dpcg_module

  !USE tea_leaf_dpcg_kernel_module
  USE tea_leaf_cheby_module
  USE tea_leaf_common_module

  USE definitions_module
  use global_mpi_module
  USE update_halo_module

  IMPLICIT NONE

  LOGICAL :: inner_use_ppcg
  INTEGER :: inner_use_ppcg_int
  REAL(KIND=8), DIMENSION(:), ALLOCATABLE :: inner_cg_alphas, inner_cg_betas
  REAL(KIND=8), DIMENSION(:), ALLOCATABLE :: inner_ch_alphas, inner_ch_betas
  REAL(KIND=8) :: eigmin, eigmax, theta

CONTAINS

SUBROUTINE tea_leaf_dpcg_init_x0(solve_time)

  IMPLICIT NONE

  REAL(KIND=8) :: solve_time

  INTEGER :: it_count, info
  INTEGER :: t
  INTEGER :: fields(NUM_FIELDS)
  REAL(KIND=8) :: halo_time,timer

  REAL(KIND=8), DIMENSION(chunk%sub_tile_dims(1), chunk%sub_tile_dims(2)) :: t2_local

  IF (.NOT. ALLOCATED(inner_cg_alphas)) THEN
    ALLOCATE(inner_cg_alphas(coarse_solve_max_iters))
    ALLOCATE(inner_cg_betas (coarse_solve_max_iters))
    ALLOCATE(inner_ch_alphas(coarse_solve_max_iters))
    ALLOCATE(inner_ch_betas (coarse_solve_max_iters))
  ENDIF

  CALL tea_leaf_dpcg_coarsen_matrix()

  ! just use CG on the first one
  inner_use_ppcg = .FALSE.
  inner_use_ppcg_int = 0

  chunk%def%t1 = 0.0_8
  CALL tea_leaf_dpcg_restrict_ZT(.TRUE.)

  !CALL tea_leaf_dpcg_local_solve(   &
  !    chunk%def%x_min, &
  !    chunk%def%x_max,                                  &
  !    chunk%def%y_min,                                  &
  !    chunk%def%y_max,                                  &
  !    halo_exchange_depth,                                  &
  !    chunk%def%t2,                               &
  !    chunk%def%t1,                               &
  !    chunk%def%def_Kx, &
  !    chunk%def%def_Ky, &
  !    chunk%def%def_di, &
  !    chunk%def%def_p,                               &
  !    chunk%def%def_r,                               &
  !    chunk%def%def_Mi,                               &
  !    chunk%def%def_w,                               &
  !    chunk%def%def_z, &
  !    chunk%def%def_sd, &
  !    coarse_solve_eps, &
  !    coarse_solve_max_iters,                          &
  !    it_count,         &
  !    0.0_8,            &
  !    inner_use_ppcg,       &
  !    inner_cg_alphas, inner_cg_betas,      &
  !    inner_ch_alphas, inner_ch_betas       &
  !    )

  IF (use_opencl_kernels) THEN
    DO t=1,tiles_per_task
      t2_local = chunk%def%t2(chunk%tiles(t)%def_tile_coords(1):chunk%tiles(t)%def_tile_coords(1)+chunk%sub_tile_dims(1)-1, &
                              chunk%tiles(t)%def_tile_coords(2):chunk%tiles(t)%def_tile_coords(2)+chunk%sub_tile_dims(2)-1)

      CALL tea_leaf_dpcg_coarse_solve_ocl(       &
            coarse_solve_eps,                   &
            coarse_solve_max_iters,             &
            it_count, theta,                    &
            inner_use_ppcg_int,                     &
            inner_cg_alphas, inner_cg_betas,    &
            inner_ch_alphas, inner_ch_betas,    &
            t2_local)

      chunk%def%t2(chunk%tiles(t)%def_tile_coords(1):chunk%tiles(t)%def_tile_coords(1)+chunk%sub_tile_dims(1)-1, &
                   chunk%tiles(t)%def_tile_coords(2):chunk%tiles(t)%def_tile_coords(2)+chunk%sub_tile_dims(2)-1) = t2_local
    ENDDO
  ENDIF

  ! add back onto the fine grid
  CALL tea_leaf_dpcg_subtract_z()

  IF (use_opencl_kernels) THEN
    DO t=1,tiles_per_task
      t2_local = chunk%def%t2(chunk%tiles(t)%def_tile_coords(1):chunk%tiles(t)%def_tile_coords(1)+chunk%sub_tile_dims(1)-1, &
                              chunk%tiles(t)%def_tile_coords(2):chunk%tiles(t)%def_tile_coords(2)+chunk%sub_tile_dims(2)-1)

      CALL tea_leaf_dpcg_subtract_u_kernel_ocl(t2_local)
    ENDDO
  ENDIF

  ! for all subsequent steps, use ppcg
  inner_use_ppcg = .TRUE.
  inner_use_ppcg_int = 1

  !CALL tea_calc_eigenvalues(inner_cg_alphas, inner_cg_betas, eigmin, eigmax, &
  !    max_iters, it_count, info)
  info = 0

  ! With jacobi preconditioner on
  eigmin = 0.01_8
  eigmax = 2.0_8

  IF (info .NE. 0) CALL report_error('tea_leaf_dpcg_init_x0', 'Error in calculating eigenvalues')

  CALL tea_calc_ch_coefs(inner_ch_alphas, inner_ch_betas, eigmin, eigmax, &
      theta, it_count)

  fields = 0
  fields(FIELD_U) = 1

  ! update the halo for u prior to recalculating the residual
  IF (profiler_on) halo_time = timer()
  CALL update_halo(fields,1)
  IF (profiler_on) solve_time = solve_time + (timer()-halo_time)

  ! calc residual again, and do initial solve
  CALL tea_leaf_calc_residual()

  CALL tea_leaf_dpcg_setup_and_solve_E(solve_time)

  CALL tea_leaf_dpcg_init_p()

END SUBROUTINE tea_leaf_dpcg_init_x0

SUBROUTINE tea_leaf_dpcg_setup_and_solve_E(solve_time)

  IMPLICIT NONE

  REAL(KIND=8) :: solve_time

  INTEGER :: it_count
  INTEGER :: t

  REAL(KIND=8), DIMENSION(chunk%sub_tile_dims(1), chunk%sub_tile_dims(2)) :: t2_local

  CALL tea_leaf_dpcg_matmul_ZTA(solve_time)
  CALL tea_leaf_dpcg_restrict_ZT(.TRUE.)

  !CALL tea_leaf_dpcg_local_solve(   &
  !    chunk%def%x_min, &
  !    chunk%def%x_max,                                  &
  !    chunk%def%y_min,                                  &
  !    chunk%def%y_max,                                  &
  !    halo_exchange_depth,                                  &
  !    chunk%def%t2,                               &
  !    chunk%def%t1,                               &
  !    chunk%def%def_Kx, &
  !    chunk%def%def_Ky, &
  !    chunk%def%def_di, &
  !    chunk%def%def_p,                               &
  !    chunk%def%def_r,                               &
  !    chunk%def%def_Mi,                               &
  !    chunk%def%def_w,                               &
  !    chunk%def%def_z, &
  !    chunk%def%def_sd, &
  !    coarse_solve_eps, &
  !    coarse_solve_max_iters,                          &
  !    it_count,         &
  !    theta,            &
  !    inner_use_ppcg,       &
  !    inner_cg_alphas, inner_cg_betas,      &
  !    inner_ch_alphas, inner_ch_betas       &
  !    )

  IF (use_opencl_kernels) THEN
    DO t=1,tiles_per_task
      t2_local = chunk%def%t2(chunk%tiles(t)%def_tile_coords(1):chunk%tiles(t)%def_tile_coords(1)+chunk%sub_tile_dims(1)-1, &
                              chunk%tiles(t)%def_tile_coords(2):chunk%tiles(t)%def_tile_coords(2)+chunk%sub_tile_dims(2)-1)

      CALL tea_leaf_dpcg_coarse_solve_ocl(       &
            coarse_solve_eps,                   &
            coarse_solve_max_iters,             &
            it_count, theta,                    &
            inner_use_ppcg_int,                     &
            inner_cg_alphas, inner_cg_betas,    &
            inner_ch_alphas, inner_ch_betas,    &
            t2_local)

      chunk%def%t2(chunk%tiles(t)%def_tile_coords(1):chunk%tiles(t)%def_tile_coords(1)+chunk%sub_tile_dims(1)-1, &
                   chunk%tiles(t)%def_tile_coords(2):chunk%tiles(t)%def_tile_coords(2)+chunk%sub_tile_dims(2)-1) = t2_local
    ENDDO
  ENDIF

  CALL tea_leaf_dpcg_prolong_Z()

END SUBROUTINE tea_leaf_dpcg_setup_and_solve_E

SUBROUTINE tea_leaf_dpcg_coarsen_matrix()

  IMPLICIT NONE
  INTEGER :: t, err

  INTEGER :: sub_tile_dx, sub_tile_dy

  INTEGER(KIND=4) :: j,k
  INTEGER(KIND=4) :: jj,j_start,j_end,kk,k_start,k_end
  REAL(KIND=8) :: tile_size
  REAL(KIND=8),dimension(chunk%sub_tile_dims(1), chunk%sub_tile_dims(2)) :: kx_local, ky_local

  chunk%def%def_Kx = 0.0_8
  chunk%def%def_Ky = 0.0_8
  chunk%def%def_di = 0.0_8

  IF (use_opencl_kernels) THEN
    DO t=1,tiles_per_task
      CALL tea_leaf_dpcg_coarsen_matrix_kernel_ocl(kx_local, ky_local)

      chunk%def%def_kx(chunk%tiles(t)%def_tile_coords(1):chunk%tiles(t)%def_tile_coords(1)+chunk%sub_tile_dims(1)-1, &
                       chunk%tiles(t)%def_tile_coords(2):chunk%tiles(t)%def_tile_coords(2)+chunk%sub_tile_dims(2)-1) = kx_local
      chunk%def%def_ky(chunk%tiles(t)%def_tile_coords(1):chunk%tiles(t)%def_tile_coords(1)+chunk%sub_tile_dims(1)-1, &
                       chunk%tiles(t)%def_tile_coords(2):chunk%tiles(t)%def_tile_coords(2)+chunk%sub_tile_dims(2)-1) = ky_local
    ENDDO
  ENDIF

  CALL MPI_Allreduce(MPI_IN_PLACE, chunk%def%def_kx, size(chunk%def%def_kx), MPI_DOUBLE_PRECISION, MPI_SUM, mpi_cart_comm, err)
  CALL MPI_Allreduce(MPI_IN_PLACE, chunk%def%def_ky, size(chunk%def%def_ky), MPI_DOUBLE_PRECISION, MPI_SUM, mpi_cart_comm, err)

  IF (use_opencl_kernels) THEN
    DO t=1,tiles_per_task
      sub_tile_dx=(chunk%tiles(t)%field%x_max-chunk%tiles(t)%field%x_min+chunk%sub_tile_dims(1))/chunk%sub_tile_dims(1)
      sub_tile_dy=(chunk%tiles(t)%field%y_max-chunk%tiles(t)%field%y_min+chunk%sub_tile_dims(2))/chunk%sub_tile_dims(2)

      DO kk=1,chunk%sub_tile_dims(2)
        k_start=chunk%tiles(t)%field%y_min+(kk-1)*sub_tile_dy
        k_end  =min(k_start+sub_tile_dy-1,chunk%tiles(t)%field%y_max)
        DO jj=1,chunk%sub_tile_dims(1)
          j_start=chunk%tiles(t)%field%x_min+(jj-1)*sub_tile_dx
          j_end  =min(j_start+sub_tile_dx-1,chunk%tiles(t)%field%x_max)
          tile_size=(j_end-j_start+1)*(k_end-k_start+1)
          chunk%def%def_di(chunk%tiles(t)%def_tile_coords(1)+jj-1, chunk%tiles(t)%def_tile_coords(2)+kk-1) = &
            tile_size + &
            chunk%def%def_kx(chunk%tiles(t)%def_tile_coords(1)+jj-1    , chunk%tiles(t)%def_tile_coords(2)+kk-1    ) + &
            chunk%def%def_ky(chunk%tiles(t)%def_tile_coords(1)+jj-1    , chunk%tiles(t)%def_tile_coords(2)+kk-1    ) + &
            chunk%def%def_kx(chunk%tiles(t)%def_tile_coords(1)+jj-1 + 1, chunk%tiles(t)%def_tile_coords(2)+kk-1    ) + &
            chunk%def%def_ky(chunk%tiles(t)%def_tile_coords(1)+jj-1    , chunk%tiles(t)%def_tile_coords(2)+kk-1 + 1)
        ENDDO
      ENDDO
    ENDDO
  ENDIF

  CALL MPI_Allreduce(MPI_IN_PLACE, chunk%def%def_di, size(chunk%def%def_di), MPI_DOUBLE_PRECISION, MPI_SUM, mpi_cart_comm, err)

  IF (use_opencl_kernels) THEN
    DO t=1,tiles_per_task
      CALL tea_leaf_dpcg_copy_reduced_coarse_grid_ocl(chunk%def%def_Kx, chunk%def%def_Ky, chunk%def%def_di)
    ENDDO
  ENDIF

END SUBROUTINE tea_leaf_dpcg_coarsen_matrix

SUBROUTINE tea_leaf_dpcg_matmul_ZTA(solve_time)

  IMPLICIT NONE

  REAL(KIND=8) :: solve_time

  INTEGER :: t, err

  INTEGER :: sub_tile_dx, sub_tile_dy

  REAL(KIND=8),dimension(chunk%sub_tile_dims(1), chunk%sub_tile_dims(2)) :: ztaz_local

  REAL(KIND=8) :: halo_time,timer

  INTEGER :: fields(NUM_FIELDS)

  ! TODO can just call from when inside C++
  !IF (use_opencl_kernels) THEN
  !  DO t=1,tiles_per_task
  !    CALL tea_leaf_dpcg_solve_z_kernel_ocl()
  !  ENDDO
  !ENDIF

  fields = 0
  fields(FIELD_Z) = 1

  IF (profiler_on) halo_time = timer()
  CALL update_halo(fields,1)
  IF (profiler_on) solve_time = solve_time + (timer()-halo_time)

  fields = 0
  fields(FIELD_P) = 1

  chunk%def%t1 = 0.0_8

  IF (use_opencl_kernels) THEN
    DO t=1,tiles_per_task
      CALL tea_leaf_dpcg_matmul_ZTA_kernel_ocl(ztaz_local)

      chunk%def%t1(chunk%tiles(t)%def_tile_coords(1):chunk%tiles(t)%def_tile_coords(1)+chunk%sub_tile_dims(1)-1, &
                   chunk%tiles(t)%def_tile_coords(2):chunk%tiles(t)%def_tile_coords(2)+chunk%sub_tile_dims(2)-1) = ztaz_local
    ENDDO
  ENDIF

END SUBROUTINE tea_leaf_dpcg_matmul_ZTA

SUBROUTINE tea_leaf_dpcg_restrict_ZT(not_init)

  IMPLICIT NONE
  LOGICAL :: not_init
  INTEGER :: t, err

  INTEGER :: sub_tile_dx, sub_tile_dy
  REAL(KIND=8),dimension(chunk%sub_tile_dims(1), chunk%sub_tile_dims(2)) :: ztr_local

  IF (use_opencl_kernels) THEN
    DO t=1,tiles_per_task
      CALL tea_leaf_dpcg_restrict_ZT_kernel_ocl(ztr_local)

      IF (not_init .EQV. .TRUE.) THEN
        chunk%def%t1(chunk%tiles(t)%def_tile_coords(1):chunk%tiles(t)%def_tile_coords(1)+chunk%sub_tile_dims(1)-1, &
                     chunk%tiles(t)%def_tile_coords(2):chunk%tiles(t)%def_tile_coords(2)+chunk%sub_tile_dims(2)-1) = &
        chunk%def%t1(chunk%tiles(t)%def_tile_coords(1):chunk%tiles(t)%def_tile_coords(1)+chunk%sub_tile_dims(1)-1, &
                     chunk%tiles(t)%def_tile_coords(2):chunk%tiles(t)%def_tile_coords(2)+chunk%sub_tile_dims(2)-1) - ztr_local
      ELSE
        chunk%def%t1(chunk%tiles(t)%def_tile_coords(1):chunk%tiles(t)%def_tile_coords(1)+chunk%sub_tile_dims(1)-1, &
                     chunk%tiles(t)%def_tile_coords(2):chunk%tiles(t)%def_tile_coords(2)+chunk%sub_tile_dims(2)-1) = - ztr_local
      ENDIF
    ENDDO
  ENDIF

  CALL MPI_Allreduce(chunk%def%t1, chunk%def%t2, size(chunk%def%t2), MPI_DOUBLE_PRECISION, MPI_SUM, mpi_cart_comm, err)

  IF (use_opencl_kernels) THEN
    DO t=1,tiles_per_task
      CALL tea_leaf_dpcg_copy_reduced_t2_ocl(chunk%def%t2)
    ENDDO
  ENDIF

END SUBROUTINE tea_leaf_dpcg_restrict_ZT

SUBROUTINE tea_leaf_dpcg_prolong_Z()

  IMPLICIT NONE

  INTEGER :: t

  INTEGER :: sub_tile_dx, sub_tile_dy

  REAL(KIND=8), DIMENSION(chunk%sub_tile_dims(1), chunk%sub_tile_dims(2)) :: t2_local

  IF (use_opencl_kernels) THEN
    DO t=1,tiles_per_task
      t2_local = chunk%def%t2(chunk%tiles(t)%def_tile_coords(1):chunk%tiles(t)%def_tile_coords(1)+chunk%sub_tile_dims(1)-1, &
                              chunk%tiles(t)%def_tile_coords(2):chunk%tiles(t)%def_tile_coords(2)+chunk%sub_tile_dims(2)-1)

      CALL tea_leaf_dpcg_prolong_Z_kernel_ocl(t2_local)
    ENDDO
  ENDIF

END SUBROUTINE tea_leaf_dpcg_prolong_Z

SUBROUTINE tea_leaf_dpcg_subtract_z()

  IMPLICIT NONE
  INTEGER :: t

  INTEGER :: sub_tile_dx, sub_tile_dy

  REAL(KIND=8), DIMENSION(chunk%sub_tile_dims(1), chunk%sub_tile_dims(2)) :: t2_local

  IF (use_opencl_kernels) THEN
    DO t=1,tiles_per_task
      t2_local = chunk%def%t2(chunk%tiles(t)%def_tile_coords(1):chunk%tiles(t)%def_tile_coords(1)+chunk%sub_tile_dims(1)-1, &
                              chunk%tiles(t)%def_tile_coords(2):chunk%tiles(t)%def_tile_coords(2)+chunk%sub_tile_dims(2)-1)

      CALL tea_leaf_dpcg_subtract_u_kernel_ocl(t2_local)
    ENDDO
  ENDIF

END SUBROUTINE tea_leaf_dpcg_subtract_z

SUBROUTINE tea_leaf_dpcg_init_p()

  IMPLICIT NONE

  INTEGER :: t

  IF (use_opencl_kernels) THEN
    DO t=1,tiles_per_task
      CALL tea_leaf_dpcg_init_p_kernel_ocl()
    ENDDO
  ENDIF

END SUBROUTINE tea_leaf_dpcg_init_p

SUBROUTINE tea_leaf_dpcg_store_r()

  IMPLICIT NONE
  INTEGER :: t

  IF (use_opencl_kernels) THEN
    DO t=1,tiles_per_task
      CALL tea_leaf_dpcg_store_r_kernel_ocl()
    ENDDO
  ENDIF

END SUBROUTINE tea_leaf_dpcg_store_r

SUBROUTINE tea_leaf_dpcg_calc_rrn(rrn)

  IMPLICIT NONE
  INTEGER :: t
  REAL(KIND=8) :: rrn, tile_rrn

  rrn = 0.0_8

  IF (use_opencl_kernels) THEN
    DO t=1,tiles_per_task
      tile_rrn = 0.0_8

      CALL tea_leaf_dpcg_calc_rrn_kernel_ocl(tile_rrn)

      rrn = rrn + tile_rrn
    ENDDO
  ENDIF

END SUBROUTINE tea_leaf_dpcg_calc_rrn

SUBROUTINE tea_leaf_dpcg_calc_p(beta)

  IMPLICIT NONE
  INTEGER :: t
  REAL(KIND=8) :: beta

  IF (use_opencl_kernels) THEN
    DO t=1,tiles_per_task
      CALL tea_leaf_dpcg_calc_p_kernel_ocl()
    ENDDO
  ENDIF

END SUBROUTINE tea_leaf_dpcg_calc_p

SUBROUTINE tea_leaf_dpcg_calc_zrnorm(rro)

  IMPLICIT NONE

  INTEGER :: t
  REAL(KIND=8) :: rro, tile_rro

  rro = 0.0_8

  IF (use_opencl_kernels) THEN
    DO t=1,tiles_per_task
      tile_rro = 0.0_8

      CALL tea_leaf_calc_2norm_kernel_ocl(2, tile_rro)

      rro = rro + tile_rro
    ENDDO
  ENDIF

END SUBROUTINE tea_leaf_dpcg_calc_zrnorm

END MODULE tea_leaf_dpcg_module

