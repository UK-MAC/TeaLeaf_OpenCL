
MODULE tea_leaf_cg_module

  USE definitions_module

  IMPLICIT NONE

CONTAINS

SUBROUTINE tea_leaf_cg_init(rro)

  IMPLICIT NONE

  INTEGER :: t
  REAL(KIND=8) :: rro, tile_rro

  rro = 0.0_8

  IF (use_opencl_kernels) THEN
    DO t=1,tiles_per_task
      tile_rro = 0.0_8

      CALL tea_leaf_cg_init_kernel_ocl(tile_rro)

      rro = rro + tile_rro
    ENDDO
  ENDIF

END SUBROUTINE tea_leaf_cg_init

SUBROUTINE tea_leaf_cg_calc_w(pw)

  IMPLICIT NONE

  INTEGER :: t
  REAL(KIND=8) :: pw, tile_pw

  pw = 0.0_08

  IF (use_opencl_kernels) THEN
    DO t=1,tiles_per_task
      tile_pw = 0.0_8

      CALL tea_leaf_cg_calc_w_kernel_ocl(chunk%tiles(t)%field%rx, &
        chunk%tiles(t)%field%ry, tile_pw)

      pw = pw + tile_pw
    ENDDO
  ENDIF

END SUBROUTINE tea_leaf_cg_calc_w

SUBROUTINE tea_leaf_cg_calc_ur(alpha, rrn)

  IMPLICIT NONE

  INTEGER :: t
  REAL(KIND=8) :: alpha, rrn, tile_rrn

  rrn = 0.0_8

  IF (use_opencl_kernels) THEN
    DO t=1,tiles_per_task
      tile_rrn = 0.0_8

      CALL tea_leaf_cg_calc_ur_kernel_ocl(alpha, tile_rrn)

      rrn = rrn + tile_rrn
    ENDDO
  ENDIF

END SUBROUTINE tea_leaf_cg_calc_ur

SUBROUTINE tea_leaf_cg_calc_p(beta)

  IMPLICIT NONE

  INTEGER :: t
  REAL(KIND=8) :: beta

  IF (use_opencl_kernels) THEN
    DO t=1,tiles_per_task
      CALL tea_leaf_cg_calc_p_kernel_ocl(beta)
    ENDDO
  ENDIF

END SUBROUTINE tea_leaf_cg_calc_p

END MODULE tea_leaf_cg_module

