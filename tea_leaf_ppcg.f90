MODULE tea_leaf_kernel_ppcg_module

use tea_leaf_kernel_module
use tea_leaf_kernel_cheby_module

IMPLICIT NONE

CONTAINS

SUBROUTINE tea_leaf_kernel_ppcg_init(x_min,             &
                           x_max,             &
                           y_min,             &
                           y_max,             &
                           u,                &
                           u0,                &
                           p,                &
                           r,            &
                           Kx,                &
                           Ky,  &
                           sd,  &
                           ch_alphas, &
                           ch_betas, &
                           rx, &
                           ry, &
                           error)
  IMPLICIT NONE

  INTEGER(KIND=4):: x_min,x_max,y_min,y_max
  REAL(KIND=8), DIMENSION(x_min-2:x_max+2,y_min-2:y_max+2) :: u
  REAL(KIND=8), DIMENSION(x_min-2:x_max+2,y_min-2:y_max+2) :: u0
  REAL(KIND=8), DIMENSION(x_min-2:x_max+2,y_min-2:y_max+2) :: p, r, sd
  REAL(KIND=8), DIMENSION(x_min-2:x_max+2,y_min-2:y_max+2) :: Kx, Ky

  INTEGER :: j,k, max_cheby_iters
  REAL(KIND=8) ::  rx, ry, error, theta, smvp
  REAL(KIND=8), DIMENSION(:) :: ch_alphas, ch_betas

  call tea_leaf_calc_residual(x_min, x_max, y_min, y_max, u, u0, r, Kx, Ky, rx, ry)
  call tea_leaf_calc_2norm_kernel(x_min, x_max, y_min, y_max, r, error)

  call tea_leaf_kernel_ppcg_inner(x_min, x_max, y_min, y_max, 1, ch_alphas, ch_betas, &
        rx, ry, u, r, Kx, Ky, sd)

!$OMP PARALLEL
!$OMP DO
  DO k=y_min,y_max
    DO j=x_min,x_max
      p(j, k) = r(j, k)
    ENDDO
  ENDDO
!$OMP END DO
!$OMP END PARALLEL

END SUBROUTINE

SUBROUTINE tea_leaf_kernel_ppcg_init_sd(x_min,             &
                           x_max,             &
                           y_min,             &
                           y_max,             &
                           r,            &
                           sd, &
                           theta)

  IMPLICIT NONE

  INTEGER(KIND=4):: x_min,x_max,y_min,y_max
  REAL(KIND=8), DIMENSION(x_min-2:x_max+2,y_min-2:y_max+2) :: r, sd
  REAL(KIND=8) :: theta

  INTEGER :: j,k

!$OMP PARALLEL
!$OMP DO
    DO k=y_min,y_max
        DO j=x_min,x_max
            sd(j, k) = r(j, k)/theta
        ENDDO
    ENDDO
!$OMP END DO
!$OMP END PARALLEL

END SUBROUTINE tea_leaf_kernel_ppcg_init_sd

SUBROUTINE tea_leaf_kernel_ppcg_inner(x_min,             &
                           x_max,             &
                           y_min,             &
                           y_max,             &
                           step, &
                           alpha, &
                           beta, &
                           rx, ry, &
                           u, &
                           r, &
                           Kx, &
                           Ky, &
                           sd)
  IMPLICIT NONE

  INTEGER(KIND=4):: x_min,x_max,y_min,y_max
  REAL(KIND=8), DIMENSION(x_min-2:x_max+2,y_min-2:y_max+2) :: u, r, Kx, Ky, sd
  INTEGER(KIND=4) :: j,k, step
  REAL(KIND=8), dimension(:) :: alpha, beta
  REAL(KIND=8) :: smvp, rx, ry

!$OMP PARALLEL
!$OMP DO
    DO k=y_min,y_max
        DO j=x_min,x_max
            smvp = (1.0_8                                      &
                + ry*(Ky(j, k+1) + Ky(j, k))                      &
                + rx*(Kx(j+1, k) + Kx(j, k)))*sd(j, k)             &
                - ry*(Ky(j, k+1)*sd(j, k+1) + Ky(j, k)*sd(j, k-1))  &
                - rx*(Kx(j+1, k)*sd(j+1, k) + Kx(j, k)*sd(j-1, k))
            r(j, k) = r(j, k) - smvp
            u(j, k) = u(j, k) + sd(j, k)
        ENDDO
    ENDDO
!$OMP END DO
!$OMP DO
    DO k=y_min,y_max
        DO j=x_min,x_max
            sd(j, k) = alpha(step)*sd(j, k) + beta(step)*r(j, k)
        ENDDO
    ENDDO
!$OMP END DO
!$OMP END PARALLEL

end SUBROUTINE

end module

