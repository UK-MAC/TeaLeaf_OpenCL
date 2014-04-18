!cROWn Copyright 2014 AWE.
!
! This file is part of TeaLeaf.
!
! TeaLeaf is free software: you can redistribute it and/or modify it under 
! the terms of the GNU General Public License as published by the 
! Free Software Foundation, either version 3 of the License, or (at your option) 
! any later version.
!
! TeaLeaf is distributed in the hope that it will be useful, but 
! WITHOUT ANY WARRANTY; without even the implied warranty of MERCHANTABILITY or 
! FITNESS FOR A PARTICULAR PURPOSE. See the GNU General Public License for more 
! details.
!
! You should have received a copy of the GNU General Public License along with 
! TeaLeaf. If not, see http://www.gnu.org/licenses/.

!>  @brief Fortran heat conduction kernel
!>  @author Michael Boulton, Wayne Gaudin
!>  @details Implicitly calculates the change in temperature using accelerated Chebyshev method

MODULE tea_leaf_kernel_cheby_module

IMPLICIT NONE

CONTAINS

subroutine tea_leaf_calc_2norm_kernel(x_min, &
                          x_max,             &
                          y_min,             &
                          y_max,             &
                          arr,               &
                          norm)

  IMPLICIT NONE

  INTEGER(KIND=4):: x_min,x_max,y_min,y_max
  REAL(KIND=8), DIMENSION(x_min-2:x_max+2,y_min-2:y_max+2) :: arr
  REAL(KIND=8) :: norm
  integer :: j, k

  norm = 0.0_8

!$OMP PARALLEL
!$OMP DO REDUCTION(+:norm)
    DO k=y_min,y_max
        DO j=x_min,x_max
            norm = norm + arr(j, k)*arr(j, k)
        ENDDO
    ENDDO
!$OMP END DO
!$OMP END PARALLEL

end subroutine tea_leaf_calc_2norm_kernel

SUBROUTINE tea_leaf_kernel_cheby_init(x_min,             &
                           x_max,             &
                           y_min,             &
                           y_max,             &
                           u,                &
                           p,                &
                           r,            &
                           u0,                &
                           w,     &
                           Kx,                &
                           Ky,  &
                           ch_alphas, &
                           ch_betas, &
                           rx, &
                           ry, &
                           theta, &
                           error)
  IMPLICIT NONE

  INTEGER(KIND=4):: x_min,x_max,y_min,y_max
  REAL(KIND=8), DIMENSION(x_min-2:x_max+2,y_min-2:y_max+2) :: u
  REAL(KIND=8), DIMENSION(x_min-2:x_max+2,y_min-2:y_max+2) :: u0
  REAL(KIND=8), DIMENSION(x_min-2:x_max+2,y_min-2:y_max+2) :: w
  REAL(KIND=8), DIMENSION(x_min-2:x_max+2,y_min-2:y_max+2) :: p, r
  REAL(KIND=8), DIMENSION(x_min-2:x_max+2,y_min-2:y_max+2) :: Kx, Ky

  INTEGER(KIND=4) :: j,k
  REAL(KIND=8) ::  rx, ry, error, theta
  REAL(KIND=8), DIMENSION(:) :: ch_alphas, ch_betas

  ! iterate once - just sets 'r' to be correct to initialise p
  call tea_leaf_kernel_cheby_iterate(x_min,&
      x_max,                       &
      y_min,                       &
      y_max,                       &
      u,                           &
      p,                 &
      r,                 &
      u0,                 &
      w,                 &
      Kx,                 &
      Ky,                 &
      ch_alphas, ch_betas, &
      rx, ry, 1)

  ! then calculate the norm
  call tea_leaf_calc_2norm_kernel(x_min,        &
      x_max,                       &
      y_min,                       &
      y_max,                       &
      r,                 &
      error)

!$OMP PARALLEL
!$OMP DO
  DO k=y_min,y_max
      DO j=x_min,x_max
          p(j, k) = r(j, k)/theta
      ENDDO
  ENDDO
!$OMP END DO
!$OMP END PARALLEL

end subroutine

SUBROUTINE tea_leaf_kernel_cheby_iterate(x_min,             &
                           x_max,             &
                           y_min,             &
                           y_max,             &
                           u,                &
                           p,                &
                           r,            &
                           u0,                &
                           w,     &
                           Kx,                &
                           Ky,  &
                           ch_alphas, &
                           ch_betas, &
                           rx, &
                           ry, &
                           step)

  IMPLICIT NONE

  INTEGER(KIND=4):: x_min,x_max,y_min,y_max
  REAL(KIND=8), DIMENSION(x_min-2:x_max+2,y_min-2:y_max+2) :: u
  REAL(KIND=8), DIMENSION(x_min-2:x_max+2,y_min-2:y_max+2) :: u0
  REAL(KIND=8), DIMENSION(x_min-2:x_max+2,y_min-2:y_max+2) :: w
  REAL(KIND=8), DIMENSION(x_min-2:x_max+2,y_min-2:y_max+2) :: p, r
  REAL(KIND=8), DIMENSION(x_min-2:x_max+2,y_min-2:y_max+2) :: Kx, Ky

  INTEGER(KIND=4) :: j,k

    REAL(KIND=8) ::  rx, ry

    REAL(KIND=8), DIMENSION(:) :: ch_alphas, ch_betas
    INTEGER(KIND=4) :: step

!$OMP PARALLEL
!$OMP DO
    DO k=y_min,y_max
        DO j=x_min,x_max
            u(j, k) = u(j, k) + p(j, k)
        ENDDO
    ENDDO
!$OMP END DO
!$OMP DO
    DO k=y_min,y_max
        DO j=x_min,x_max
            w(j, k) = (1.0_8                                      &
                + ry*(Ky(j, k+1) + Ky(j, k))                      &
                + rx*(Kx(j+1, k) + Kx(j, k)))*u(j, k)             &
                - ry*(Ky(j, k+1)*u(j, k+1) + Ky(j, k)*u(j, k-1))  &
                - rx*(Kx(j+1, k)*u(j+1, k) + Kx(j, k)*u(j-1, k))
            r(j, k) = u0(j, k) - w(j, k)
            p(j, k) = ch_alphas(step)*p(j, k) + ch_betas(step)*r(j, k)
        ENDDO
    ENDDO
!$OMP END DO
!$OMP END PARALLEL

END SUBROUTINE tea_leaf_kernel_cheby_iterate

SUBROUTINE tea_leaf_kernel_cheby_copy_u(x_min,             &
                           x_max,             &
                           y_min,             &
                           y_max,             &
                           u0, u)
  IMPLICIT NONE

  INTEGER(KIND=4):: x_min,x_max,y_min,y_max
  REAL(KIND=8), DIMENSION(x_min-2:x_max+2,y_min-2:y_max+2) :: u, u0
  INTEGER(KIND=4) :: j,k

!$OMP PARALLEL
!$OMP DO
    DO k=y_min,y_max
        DO j=x_min,x_max
            u0(j, k) = u(j, k)
        ENDDO
    ENDDO
!$OMP END DO
!$OMP END PARALLEL

end subroutine

end module
