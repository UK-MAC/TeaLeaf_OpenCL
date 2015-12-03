!Crown Copyright 2014 AWE.
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

!>  @brief  Allocates the data for each mesh chunk
!>  @author David Beckingsale, Wayne Gaudin
!>  @details The data fields for the mesh chunk are allocated based on the mesh
!>  size.

SUBROUTINE build_field()

  USE tea_module

  IMPLICIT NONE

  INTEGER :: j,k, t

  ! TODO won't be hardcoded in future
  INTEGER :: n_levels=2
  ! TODO make a 3d array?
  INTEGER :: tile_bounds(4, 2)

  IF (use_opencl_kernels) THEN
    DO t=1,tiles_per_task
      ! this is the number of cells
      tile_bounds(1,1) = chunk%tiles(t)%x_cells
      tile_bounds(2,1) = chunk%tiles(t)%y_cells

      ! this is the LOCAL size of this tile's portion of the GLOBAL coarser
      ! array, where the GLOBAL size of the coarser array is just the size of the
      ! next coarsest grid
      tile_bounds(3,1) = chunk%sub_tile_dims(1)
      tile_bounds(4,1) = chunk%sub_tile_dims(2)

      ! again for coarse grid
      tile_bounds(1,2) = chunk%def%x_cells
      tile_bounds(2,2) = chunk%def%y_cells
      ! This is a bit irrelevant, will never be used with 2 level scheme
      tile_bounds(3,2) = 1
      tile_bounds(4,2) = 1

      CALL initialise_ocl(tile_bounds, n_levels)
    ENDDO
  ENDIF

  ALLOCATE(chunk%def%t1(                                                            &
    chunk%def%x_min - halo_exchange_depth:chunk%def%x_max + halo_exchange_depth,    &
    chunk%def%y_min - halo_exchange_depth:chunk%def%y_max + halo_exchange_depth))
  ALLOCATE(chunk%def%t2(                                                            &
    chunk%def%x_min - halo_exchange_depth:chunk%def%x_max + halo_exchange_depth,    &
    chunk%def%y_min - halo_exchange_depth:chunk%def%y_max + halo_exchange_depth))
  ALLOCATE(chunk%def%def_kx(                                                            &
    chunk%def%x_min - halo_exchange_depth:chunk%def%x_max + halo_exchange_depth,    &
    chunk%def%y_min - halo_exchange_depth:chunk%def%y_max + halo_exchange_depth))
  ALLOCATE(chunk%def%def_ky(                                                            &
    chunk%def%x_min - halo_exchange_depth:chunk%def%x_max + halo_exchange_depth,    &
    chunk%def%y_min - halo_exchange_depth:chunk%def%y_max + halo_exchange_depth))
  ALLOCATE(chunk%def%def_di(                                                            &
    chunk%def%x_min - halo_exchange_depth:chunk%def%x_max + halo_exchange_depth,    &
    chunk%def%y_min - halo_exchange_depth:chunk%def%y_max + halo_exchange_depth))

  DO k=chunk%def%y_min - halo_exchange_depth,chunk%def%y_max + halo_exchange_depth
    DO j=chunk%def%x_min - halo_exchange_depth,chunk%def%x_max + halo_exchange_depth
      chunk%def%t1(j, k) = 0.0
      chunk%def%t2(j, k) = 0.0
      chunk%def%def_kx(j, k) = 0.0
      chunk%def%def_ky(j, k) = 0.0
      chunk%def%def_di(j, k) = 0.0
    ENDDO
  ENDDO

END SUBROUTINE build_field
