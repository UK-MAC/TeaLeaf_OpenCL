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
!>  @author David Beckingsale, Wayne Gaudin
!>  @details Implicitly calculates the change in temperature using a Jacobi iteration

MODULE tea_leaf_kernel_module

CONTAINS

SUBROUTINE tea_leaf_kernel_init(x_min,             &
                           x_max,             &
                           y_min,             &
                           y_max,             &
                           celldx,            &
                           celldy,            &
                           volume,            &
                           density,           &
                           energy,            &
                           u0,                &
                           u1,                &
                           un,                &
                           heat_capacity,     &
                           Kx_tmp,            &
                           Ky_tmp,            &
                           Kx,                &
                           Ky,                &
                           coef)

! clover_module used for coefficient constants
  USE clover_module
  USE report_module

  IMPLICIT NONE

  INTEGER(KIND=4):: x_min,x_max,y_min,y_max
  REAL(KIND=8), DIMENSION(x_min-2:x_max+2) :: celldx
  REAL(KIND=8), DIMENSION(y_min-2:y_max+2) :: celldy
  REAL(KIND=8), DIMENSION(x_min-2:x_max+2,y_min-2:y_max+2) :: volume
  REAL(KIND=8), DIMENSION(x_min-2:x_max+2,y_min-2:y_max+2) :: density
  REAL(KIND=8), DIMENSION(x_min-2:x_max+2,y_min-2:y_max+2) :: energy
  REAL(KIND=8), DIMENSION(x_min-2:x_max+3,y_min-2:y_max+3) :: u0
  REAL(KIND=8), DIMENSION(x_min-2:x_max+3,y_min-2:y_max+3) :: heat_capacity
  REAL(KIND=8), DIMENSION(x_min-2:x_max+3,y_min-2:y_max+3) :: Kx_tmp
  REAL(KIND=8), DIMENSION(x_min-2:x_max+3,y_min-2:y_max+3) :: Ky_tmp
  REAL(KIND=8), DIMENSION(x_min-2:x_max+3,y_min-2:y_max+3) :: Kx
  REAL(KIND=8), DIMENSION(x_min-2:x_max+3,y_min-2:y_max+3) :: Ky
  REAL(KIND=8), DIMENSION(x_min-2:x_max+2,y_min-2:y_max+2) :: u1
  REAL(KIND=8), DIMENSION(x_min-2:x_max+3,y_min-2:y_max+3) :: un

  INTEGER(KIND=4) :: coef

  INTEGER(KIND=4) :: j,k,n


! CALC DIFFUSION COEFFICIENT
!$OMP PARALLEL
  IF(coef .EQ. RECIP_CONDUCTIVITY) THEN
!$OMP DO 
    DO k=y_min-1,y_max+1
      DO j=x_min-1,x_max+1
         Kx_tmp(j  ,k  )=1.0_8/density(j  ,k  )
         Kx_tmp(j+1,k  )=1.0_8/density(j+1,k  )
         Ky_tmp(j  ,k  )=1.0_8/density(j  ,k  )
         Ky_tmp(j  ,k+1)=1.0_8/density(j  ,k+1)
      ENDDO
    ENDDO
!$OMP END DO
  ELSE IF(coef .EQ. CONDUCTIVITY) THEN
!$OMP DO
    DO k=y_min-1,y_max+1
      DO j=x_min-1,x_max+1
         Kx_tmp(j  ,k  )=density(j  ,k  )
         Kx_tmp(j+1,k  )=density(j+1,k  )
         Ky_tmp(j  ,k  )=density(j  ,k  )
         Ky_tmp(j  ,k+1)=density(j  ,k+1)
      ENDDO
    ENDDO
!$OMP END DO
  ELSE
    CALL report_error('tea_leaf', 'unknown coefficient option')
  ENDIF

!$OMP DO
  DO k=y_min,y_max+1
    DO j=x_min,x_max+1
         Kx(j,k)=(Kx_tmp(j-1,k  )+Kx_tmp(j,k))/(2.0_8*Kx_tmp(j-1,k  )*Kx_tmp(j,k))
         Ky(j,k)=(Ky_tmp(j  ,k-1)+Ky_tmp(j,k))/(2.0_8*Ky_tmp(j,  k-1)*Ky_tmp(j,k))
    ENDDO
  ENDDO
!$OMP END DO

!$OMP DO 
  DO k=y_min-1, y_max+1
    DO j=x_min-1, x_max+1
      u0(j,k) =  energy(j,k) * density(j,k)
    ENDDO
  ENDDO
!$OMP END DO

  ! INITIAL GUESS
!$OMP DO
  DO k=y_min-1, y_max+1
    DO j=x_min-1, x_max+1
      u1(j,k) = u0(j,k)
    ENDDO
  ENDDO
!$OMP END DO
!$OMP END PARALLEL


END SUBROUTINE tea_leaf_kernel_init

SUBROUTINE tea_leaf_kernel_solve(x_min,       &
                           x_max,             &
                           y_min,             &
                           y_max,             &
                           rx,                &
                           ry,                &
                           Kx,                &
                           Ky,                &
                           error,             &
                           u0,                &
                           u1,                &
                           un)


  IMPLICIT NONE

  INTEGER(KIND=4):: x_min,x_max,y_min,y_max
  REAL(KIND=8), DIMENSION(x_min-2:x_max+3,y_min-2:y_max+3) :: u0, un
  REAL(KIND=8), DIMENSION(x_min-2:x_max+2,y_min-2:y_max+2) :: u1
  REAL(KIND=8), DIMENSION(x_min-2:x_max+3,y_min-2:y_max+3) :: Kx
  REAL(KIND=8), DIMENSION(x_min-2:x_max+3,y_min-2:y_max+3) :: Ky

  REAL(KIND=8) :: ry,rx, error

  INTEGER(KIND=4) :: j,k

  error = 0.0_8

!$OMP PARALLEL
!$OMP DO
    DO k=y_min-1, y_max+1
      DO j=x_min-1, x_max+1
        un(j,k) = u1(j,k)
      ENDDO
    ENDDO
!$OMP END DO

!$OMP DO REDUCTION(MAX:error)
    DO k=y_min, y_max
      DO j=x_min, x_max
        u1(j,k) = (u0(j,k) + rx*(Kx(j+1,k  )*un(j+1,k  ) + Kx(j  ,k  )*un(j-1,k  )) &
                           + ry*(Ky(j  ,k+1)*un(j  ,k+1) + Ky(j  ,k  )*un(j  ,k-1))) &
                             /(1.0_8 &
                                + rx*2.0_8*(0.5_8*(Kx(j,k)+Kx(j+1,k))) &
                                + ry*2.0_8*(0.5_8*(Ky(j,k)+Ky(j,k+1))))

        error = MAX(error, ABS(u1(j,k)-un(j,k)))
      ENDDO
    ENDDO
!$OMP END DO
!$OMP END PARALLEL

END SUBROUTINE tea_leaf_kernel_solve

SUBROUTINE tea_leaf_kernel_init_cg_fortran(x_min,             &
                           x_max,             &
                           y_min,             &
                           y_max,             &
                           density,           &
                           energy,            &
                           u,                 &
                           p,           & ! 1
                           r,           & ! 2
                           Mi,          & ! 3
                           w,           & ! 4
                           z,           & ! 5
                           Kx,          & ! 6
                           Ky,          & ! 7
                           rx,          &
                           ry,          &
                           rro,         &
                           coef)

  USE clover_module
  USE report_module
  IMPLICIT NONE

  INTEGER(KIND=4):: x_min,x_max,y_min,y_max
  REAL(KIND=8), DIMENSION(x_min-2:x_max+2,y_min-2:y_max+2) :: density
  REAL(KIND=8), DIMENSION(x_min-2:x_max+2,y_min-2:y_max+2) :: energy
  REAL(KIND=8), DIMENSION(x_min-2:x_max+2,y_min-2:y_max+2) :: u
  REAL(KIND=8), DIMENSION(x_min-2:x_max+2,y_min-2:y_max+2) :: p
  REAL(KIND=8), DIMENSION(x_min-2:x_max+2,y_min-2:y_max+2) :: r
  REAL(KIND=8), DIMENSION(x_min-2:x_max+2,y_min-2:y_max+2) :: Mi
  REAL(KIND=8), DIMENSION(x_min-2:x_max+2,y_min-2:y_max+2) :: w
  REAL(KIND=8), DIMENSION(x_min-2:x_max+2,y_min-2:y_max+2) :: z
  REAL(KIND=8), DIMENSION(x_min-2:x_max+2,y_min-2:y_max+2) :: Kx
  REAL(KIND=8), DIMENSION(x_min-2:x_max+2,y_min-2:y_max+2) :: Ky

  INTEGER(KIND=4) :: coef
  INTEGER(KIND=4) :: j,k,n

  REAL(kind=8) :: rro
  REAL(KIND=8) ::  rx, ry

  rro = 0.0_8
  p = 0.0_8
  r = 0.0_8

  IF(coef .nE. RECIP_CONDUCTIVITY .and. coef .ne. conductivity) THEN
    CALL report_error('tea_leaf', 'unknown coefficient option')
  endif

!$OMP PARALLEL
!$OMP DO 
  DO k=y_min-2, y_max+2
    DO j=x_min-2, x_max+2
      u(j,k) = energy(j,k)*density(j,k)
    ENDDO
  ENDDO
!$OMP END DO

  IF(coef .EQ. RECIP_CONDUCTIVITY) THEN
!$OMP DO 
    ! use w as temp val
    DO k=y_min-1,y_max+1
      DO j=x_min-1,x_max+1
         w(j  ,k  )=1.0_8/density(j  ,k  )
      ENDDO
    ENDDO
!$OMP END DO
  ELSE IF(coef .EQ. CONDUCTIVITY) THEN
!$OMP DO
    DO k=y_min-1,y_max+1
      DO j=x_min-1,x_max+1
         w(j  ,k  )=density(j  ,k  )
      ENDDO
    ENDDO
!$OMP END DO
  ENDIF

!$OMP DO
   DO k=y_min,y_max+1
     DO j=x_min,x_max+1
          Kx(j,k)=(w(j-1,k  ) + w(j,k))/(2.0_8*w(j-1,k  )*w(j,k))
          Ky(j,k)=(w(j  ,k-1) + w(j,k))/(2.0_8*w(j  ,k-1)*w(j,k))
     ENDDO
   ENDDO
!$OMP END DO

!$OMP DO REDUCTION(+:rro)
    DO k=y_min,y_max
        DO j=x_min,x_max
            w(j, k) = (1.0                                      &
                + ry*(Ky(j, k+1) + Ky(j, k))                      &
                + rx*(Kx(j+1, k) + Kx(j, k)))*u(j, k)             &
                - ry*(Ky(j, k+1)*u(j, k+1) + Ky(j, k)*u(j, k-1))  &
                - rx*(Kx(j+1, k)*u(j+1, k) + Kx(j, k)*u(j-1, k))

            r(j, k) = u(j, k) - w(j, k)

            ! inverse diagonal used as preconditioner
            Mi(j, k) = (1.0                                      &
                + ry*(Ky(j, k+1) + Ky(j, k))                      &
                + rx*(Kx(j+1, k) + Kx(j, k)))
            Mi(j, k) = 1.0_8/Mi(j, k)

            z(j, k) = Mi(j, k)*r(j, k)
            p(j, k) = z(j, k)

            rro = rro + r(j, k)*z(j, k);
        ENDDO
    ENDDO
!$OMP END DO
!$OMP END PARALLEL

END SUBROUTINE tea_leaf_kernel_init_cg_fortran

SUBROUTINE tea_leaf_kernel_solve_cg_fortran_calc_w(x_min,             &
                           x_max,             &
                           y_min,             &
                           y_max,             &
                           p,            &
                           w,     &
                           Kx,  &
                           Ky,            &
                           rx, &
                           ry, &
                           pw)

  IMPLICIT NONE

  INTEGER(KIND=4):: x_min,x_max,y_min,y_max
  REAL(KIND=8), DIMENSION(x_min-2:x_max+2,y_min-2:y_max+2) :: p
  REAL(KIND=8), DIMENSION(x_min-2:x_max+2,y_min-2:y_max+2) :: w
  REAL(KIND=8), DIMENSION(x_min-2:x_max+2,y_min-2:y_max+2) :: Kx
  REAL(KIND=8), DIMENSION(x_min-2:x_max+2,y_min-2:y_max+2) :: Ky

    REAL(KIND=8) ::  rx, ry

    INTEGER(KIND=4) :: j,k,n
    REAL(kind=8) :: pw

    pw = 0.0

!$OMP PARALLEL
!$OMP DO REDUCTION(+:pw)
    DO k=y_min,y_max
        DO j=x_min,x_max
            w(j, k) = (1.0                                      &
                + ry*(Ky(j, k+1) + Ky(j, k))                      &
                + rx*(Kx(j+1, k) + Kx(j, k)))*p(j, k)             &
                - ry*(Ky(j, k+1)*p(j, k+1) + Ky(j, k)*p(j, k-1))  &
                - rx*(Kx(j+1, k)*p(j+1, k) + Kx(j, k)*p(j-1, k))

            pw = pw + w(j, k)*p(j, k)
        ENDDO
    ENDDO
!$OMP END DO
!$OMP END PARALLEL

END SUBROUTINE tea_leaf_kernel_solve_cg_fortran_calc_w

SUBROUTINE tea_leaf_kernel_solve_cg_fortran_calc_ur(x_min,             &
                           x_max,             &
                           y_min,             &
                           y_max,             &
                           u,                &
                           p,            &
                           r,            &
                           Mi,                &
                           w,     &
                           z,     &
                           alpha, &
                           rrn)

  IMPLICIT NONE

  INTEGER(KIND=4):: x_min,x_max,y_min,y_max
  REAL(KIND=8), DIMENSION(x_min-2:x_max+2,y_min-2:y_max+2) :: u
  REAL(KIND=8), DIMENSION(x_min-2:x_max+2,y_min-2:y_max+2) :: p
  REAL(KIND=8), DIMENSION(x_min-2:x_max+2,y_min-2:y_max+2) :: r
  REAL(KIND=8), DIMENSION(x_min-2:x_max+2,y_min-2:y_max+2) :: Mi
  REAL(KIND=8), DIMENSION(x_min-2:x_max+2,y_min-2:y_max+2) :: w
  REAL(KIND=8), DIMENSION(x_min-2:x_max+2,y_min-2:y_max+2) :: z

    INTEGER(KIND=4) :: j,k,n
    REAL(kind=8) :: alpha, rrn

    rrn = 0.0

!$OMP PARALLEL
!$OMP DO REDUCTION(+:rrn)
    DO k=y_min,y_max
        DO j=x_min,x_max
            u(j, k) = u(j, k) + alpha*p(j, k)
            r(j, k) = r(j, k) - alpha*w(j, k)
            z(j, k) = Mi(j, k)*r(j, k)

            rrn = rrn + r(j, k)*z(j, k)
        ENDDO
    ENDDO
!$OMP END DO
!$OMP END PARALLEL

END SUBROUTINE tea_leaf_kernel_solve_cg_fortran_calc_ur

SUBROUTINE tea_leaf_kernel_solve_cg_fortran_calc_p(x_min,             &
                           x_max,             &
                           y_min,             &
                           y_max,             &
                           u,                &
                           p,            &
                           r,            &
                           z,     &
                           beta)

  IMPLICIT NONE

  INTEGER(KIND=4):: x_min,x_max,y_min,y_max
  REAL(KIND=8), DIMENSION(x_min-2:x_max+2,y_min-2:y_max+2) :: u
  REAL(KIND=8), DIMENSION(x_min-2:x_max+2,y_min-2:y_max+2) :: p
  REAL(KIND=8), DIMENSION(x_min-2:x_max+2,y_min-2:y_max+2) :: r
  REAL(KIND=8), DIMENSION(x_min-2:x_max+2,y_min-2:y_max+2) :: z

    REAL(kind=8) :: error

    INTEGER(KIND=4) :: j,k,n
    REAL(kind=8) :: beta

!$OMP PARALLEL
!$OMP DO
    DO k=y_min,y_max
        DO j=x_min,x_max
            p(j, k) = z(j, k) + beta*p(j, k)
        ENDDO
    ENDDO
!$OMP END DO
!$OMP END PARALLEL

END SUBROUTINE tea_leaf_kernel_solve_cg_fortran_calc_p

! Finalise routine is used by both implementations
SUBROUTINE tea_leaf_kernel_finalise(x_min,    &
                           x_max,             &
                           y_min,             &
                           y_max,             &
                           energy,            &
                           density,           &
                           u)

  IMPLICIT NONE

  INTEGER(KIND=4):: x_min,x_max,y_min,y_max
  REAL(KIND=8), DIMENSION(x_min-2:x_max+2,y_min-2:y_max+2) :: u
  REAL(KIND=8), DIMENSION(x_min-2:x_max+2,y_min-2:y_max+2) :: energy
  REAL(KIND=8), DIMENSION(x_min-2:x_max+2,y_min-2:y_max+2) :: density

  INTEGER(KIND=4) :: j,k

!$OMP PARALLEL DO 
  DO k=y_min, y_max
    DO j=x_min, x_max
      energy(j,k) = u(j,k) / density(j,k)
    ENDDO
  ENDDO
!$OMP END PARALLEL DO

END SUBROUTINE tea_leaf_kernel_finalise

END MODULE tea_leaf_kernel_module

