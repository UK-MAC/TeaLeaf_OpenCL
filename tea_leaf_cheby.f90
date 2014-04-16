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

subroutine calc_bb_kernel(x_min, &
                           x_max,             &
                           y_min,             &
                           y_max,             &
b, bb)
  IMPLICIT NONE
  INTEGER(KIND=4):: x_min,x_max,y_min,y_max
  REAL(KIND=8), DIMENSION(x_min-2:x_max+2,y_min-2:y_max+2) :: b
  REAL(KIND=8) :: bb
  integer :: j, k
!$OMP PARALLEL
!$OMP DO REDUCTION(+:bb)
    DO k=y_min,y_max
        DO j=x_min,x_max
            bb = bb + b(j, k)*b(j, k)
        ENDDO
    ENDDO
!$OMP END DO
!$OMP END PARALLEL
end subroutine

SUBROUTINE tl_calc_resid(x_min,             &
                           x_max,             &
                           y_min,             &
                           y_max,             &
                           rx, &
                           ry, &
                           u,                &
                           u0,                &
                           w,     &
                           r, &
                           Kx,                &
                           Ky,  &
                           error)
  IMPLICIT NONE

  INTEGER(KIND=4):: x_min,x_max,y_min,y_max
  REAL(KIND=8), DIMENSION(x_min-2:x_max+2,y_min-2:y_max+2) :: u
  REAL(KIND=8), DIMENSION(x_min-2:x_max+2,y_min-2:y_max+2) :: u0
  REAL(KIND=8), DIMENSION(x_min-2:x_max+2,y_min-2:y_max+2) :: w
  REAL(KIND=8), DIMENSION(x_min-2:x_max+2,y_min-2:y_max+2) :: Kx, Ky
  REAL(KIND=8), DIMENSION(x_min-2:x_max+2,y_min-2:y_max+2) :: r

  INTEGER(KIND=4) :: j,k,n
  real(kind=8) :: error
  real(kind=8) :: rx, ry

  error = 0.0

!$OMP PARALLEL PRIVATE(j)
!$OMP DO reduction(+:error)
  DO k=y_min,y_max
    DO j=x_min,x_max
      w(j, k) = (1.0_8                                      &
          + ry*(Ky(j, k+1) + Ky(j, k))                      &
          + rx*(Kx(j+1, k) + Kx(j, k)))*u(j, k)             &
          - ry*(Ky(j, k+1)*u(j, k+1) + Ky(j, k)*u(j, k-1))  &
          - rx*(Kx(j+1, k)*u(j+1, k) + Kx(j, k)*u(j-1, k))
      r(j, k) = u0(j, k) - w(j, k)

      error = error + r(j, k)*r(j, k)
    ENDDO
  ENDDO
!$OMP END DO
!$OMP END PARALLEL

end subroutine

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
                           rx, &
                           ry, &
                           theta)
  IMPLICIT NONE

  INTEGER(KIND=4):: x_min,x_max,y_min,y_max
  REAL(KIND=8), DIMENSION(x_min-2:x_max+2,y_min-2:y_max+2) :: u
  REAL(KIND=8), DIMENSION(x_min-2:x_max+2,y_min-2:y_max+2) :: u0
  REAL(KIND=8), DIMENSION(x_min-2:x_max+2,y_min-2:y_max+2) :: w
  REAL(KIND=8), DIMENSION(x_min-2:x_max+2,y_min-2:y_max+2) :: p, r
  REAL(KIND=8), DIMENSION(x_min-2:x_max+2,y_min-2:y_max+2) :: Kx, Ky

  INTEGER(KIND=4) :: j,k,n
  REAL(KIND=8) ::  rx, ry, error, theta

  ! calculate residual - just sets 'r' to be correct to initialise p
  call tl_calc_resid(x_min, x_max, y_min, y_max, rx, ry, &
      u, u0, w, r, Kx, Ky, error)

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

SUBROUTINE tea_leaf_kernel_solve_cheby_fortran(x_min,             &
                           x_max,             &
                           y_min,             &
                           y_max,             &
                           rx, &
                           ry, &
                           x,                &
                           p,                &
                           r,            &
                           u0,                &
                           w,     &
                           Kx,                &
                           Ky,  &
                           rhoold, &
                           eigmin,  &
                           eigmax, &
                           step, &
                           error)

  IMPLICIT NONE

  INTEGER(KIND=4):: x_min,x_max,y_min,y_max
  REAL(KIND=8), DIMENSION(x_min-2:x_max+2,y_min-2:y_max+2) :: x
  REAL(KIND=8), DIMENSION(x_min-2:x_max+2,y_min-2:y_max+2) :: u0
  REAL(KIND=8), DIMENSION(x_min-2:x_max+2,y_min-2:y_max+2) :: w
  REAL(KIND=8), DIMENSION(x_min-2:x_max+2,y_min-2:y_max+2) :: p, r
  REAL(KIND=8), DIMENSION(x_min-2:x_max+2,y_min-2:y_max+2) :: Kx, Ky

  INTEGER(KIND=4) :: j,k,n, step

    REAL(KIND=8) ::  rx, ry

    real(kind=8) :: error, eigmin, eigmax, rhoold, rhonew

    real(kind=8) :: alpha, beta, sigma
    real(kind=8) :: theta, delta

    error = 0.0

!$OMP PARALLEL
!$OMP DO
    DO k=y_min,y_max
        DO j=x_min,x_max
            x(j, k) = x(j, k) + p(j, k)
        ENDDO
    ENDDO
!$OMP END DO
!$OMP DO REDUCTION(+:error)
    DO k=y_min,y_max
        DO j=x_min,x_max
            w(j, k) = (1.0_8                                      &
                + ry*(Ky(j, k+1) + Ky(j, k))                      &
                + rx*(Kx(j+1, k) + Kx(j, k)))*x(j, k)             &
                - ry*(Ky(j, k+1)*x(j, k+1) + Ky(j, k)*x(j, k-1))  &
                - rx*(Kx(j+1, k)*x(j+1, k) + Kx(j, k)*x(j-1, k))
            r(j, k) = u0(j, k) - w(j, k)
            error = error + r(j, k)*r(j, k)
            p(j, k) = alpha*p(j, k) + beta*r(j, k)
        ENDDO
    ENDDO
!$OMP END DO
!$OMP END PARALLEL

END SUBROUTINE tea_leaf_kernel_solve_cheby_fortran

end module
