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

!>  @brief Holds the high level Fortran data types
!>  @author David Beckingsale, Wayne Gaudin
!>  @details The high level data types used to store the mesh and field data
!>  are defined here.
!>
!>  Also the global variables used for defining the input and controlling the
!>  scheme are defined here.

MODULE definitions_module

   USE data_module
   
   IMPLICIT NONE

   TYPE state_type
      LOGICAL            :: defined

      REAL(KIND=8)       :: density          &
                           ,energy           &
                           ,xvel             &
                           ,yvel

      INTEGER            :: geometry

      REAL(KIND=8)       :: xmin               &
                           ,xmax               &
                           ,ymin               &
                           ,ymax               &
                           ,radius
   END TYPE state_type

   TYPE(state_type), ALLOCATABLE             :: states(:)
   INTEGER                                   :: number_of_states

   TYPE grid_type
     REAL(KIND=8)       :: xmin            &
                          ,ymin            &
                          ,xmax            &
                          ,ymax
                     
     INTEGER            :: x_cells              &
                          ,y_cells
   END TYPE grid_type

   INTEGER      :: step

   LOGICAL      :: advect_x

   INTEGER      :: error_condition

   INTEGER      :: test_problem
   LOGICAL      :: complete

   LOGICAL      :: use_fortran_kernels
   LOGICAL      :: use_C_kernels
   LOGICAL      :: use_OA_kernels
   LOGICAL      :: use_opencl_kernels
   LOGICAL      :: use_Tealeaf
   LOGICAL      :: use_Hydro
   LOGICAL      :: tl_use_chebyshev
   LOGICAL      :: tl_use_cg
   LOGICAL      :: tl_use_ppcg
   LOGICAL      :: tl_use_jacobi
   INTEGER      :: max_iters
   REAL(KIND=8) :: eps
   INTEGER      :: coefficient

   ! for chebyshev solver - whether to run cg until a certain error (tl_ch_eps)
   ! is reached, or for a certain number of steps (tl_ch_cg_presteps)
   LOGICAL      :: tl_ch_cg_errswitch
   ! error to run cg to if tl_ch_cg_errswitch is set
   REAL(KIND=8) :: tl_ch_cg_epslim
   ! number of steps of cg to run to before switching to ch if tl_ch_cg_errswitch not set
   INTEGER      :: tl_ch_cg_presteps

   LOGICAL      :: use_vector_loops ! Some loops work better in serial depending on the hardware

   LOGICAL      :: profiler_on ! Internal code profiler to make comparisons across systems easier

   ! Profile execution time per iteration of lienar solver.
   ! Want to know the time taken per step, but turning profiling on
   ! interferes with GPU based solvers as profiling requires
   ! waiting for each kernel to finish execution
   LOGICAL      :: profile_solver
                                

   TYPE profiler_type
     REAL(KIND=8)       :: timestep        &
                          ,acceleration    &
                          ,PdV             &
                          ,cell_advection  &
                          ,mom_advection   &
                          ,viscosity       &
                          ,ideal_gas       &
                          ,visit           &
                          ,summary         &
                          ,reset           &
                          ,revert          &
                          ,flux            &
                          ,tea             &
                          ,set_field       &
                          ,halo_exchange
                     
   END TYPE profiler_type
   TYPE(profiler_type)  :: profiler

   REAL(KIND=8) :: end_time

   INTEGER      :: end_step

   REAL(KIND=8) :: dtold          &
                  ,dt             &
                  ,time           &
                  ,dtinit         &
                  ,dtmin          &
                  ,dtmax          &
                  ,dtrise         &
                  ,dtu_safe       &
                  ,dtv_safe       &
                  ,dtc_safe       &
                  ,dtdiv_safe     &
                  ,dtc            &
                  ,dtu            &
                  ,dtv            &
                  ,dtdiv

   INTEGER      :: visit_frequency   &
                  ,summary_frequency

   INTEGER         :: jdt,kdt

   TYPE field_type
     REAL(KIND=8),    DIMENSION(:,:), ALLOCATABLE :: density0,density1
     REAL(KIND=8),    DIMENSION(:,:), ALLOCATABLE :: energy0,energy1
     REAL(KIND=8),    DIMENSION(:,:), ALLOCATABLE :: pressure
     REAL(KIND=8),    DIMENSION(:,:), ALLOCATABLE :: viscosity
     REAL(KIND=8),    DIMENSION(:,:), ALLOCATABLE :: soundspeed
     REAL(KIND=8),    DIMENSION(:,:), ALLOCATABLE :: xvel0,xvel1
     REAL(KIND=8),    DIMENSION(:,:), ALLOCATABLE :: yvel0,yvel1
     REAL(KIND=8),    DIMENSION(:,:), ALLOCATABLE :: vol_flux_x,mass_flux_x
     REAL(KIND=8),    DIMENSION(:,:), ALLOCATABLE :: vol_flux_y,mass_flux_y
     REAL(KIND=8),    DIMENSION(:,:), ALLOCATABLE :: u, u0
     REAL(KIND=8),    DIMENSION(:,:), ALLOCATABLE :: work_array1 !node_flux, stepbymass, volume_change, pre_vol
     REAL(KIND=8),    DIMENSION(:,:), ALLOCATABLE :: work_array2 !node_mass_post, post_vol
     REAL(KIND=8),    DIMENSION(:,:), ALLOCATABLE :: work_array3 !node_mass_pre,pre_mass
     REAL(KIND=8),    DIMENSION(:,:), ALLOCATABLE :: work_array4 !advec_vel, post_mass
     REAL(KIND=8),    DIMENSION(:,:), ALLOCATABLE :: work_array5 !mom_flux, advec_vol
     REAL(KIND=8),    DIMENSION(:,:), ALLOCATABLE :: work_array6 !pre_vol, post_ener
     REAL(KIND=8),    DIMENSION(:,:), ALLOCATABLE :: work_array7 !post_vol, ener_flux

     INTEGER         :: left            &
                       ,right           &
                       ,bottom          &
                       ,top             &
                       ,left_boundary   &
                       ,right_boundary  &
                       ,bottom_boundary &
                       ,top_boundary

     INTEGER         :: x_min  &
                       ,y_min  &
                       ,x_max  &
                       ,y_max

     REAL(KIND=8), DIMENSION(:),   ALLOCATABLE :: cellx    &
                                                 ,celly    &
                                                 ,vertexx  &
                                                 ,vertexy  &
                                                 ,celldx   &
                                                 ,celldy   &
                                                 ,vertexdx &
                                                 ,vertexdy

     REAL(KIND=8), DIMENSION(:,:), ALLOCATABLE :: volume  &
                                                 ,xarea   &
                                                 ,yarea

   END TYPE field_type
   
   TYPE chunk_type

     INTEGER         :: task   !mpi task

     INTEGER         :: chunk_neighbours(4) ! Chunks, not tasks, so we can overload in the future

     ! Idealy, create an array to hold the buffers for each field so a commuincation only needs
     !  one send and one receive per face, rather than per field.
     ! If chunks are overloaded, i.e. more chunks than tasks, might need to pack for a task to task comm 
     !  rather than a chunk to chunk comm. See how performance is at high core counts before deciding
     REAL(KIND=8),ALLOCATABLE:: left_rcv_buffer(:),right_rcv_buffer(:),bottom_rcv_buffer(:),top_rcv_buffer(:)
     REAL(KIND=8),ALLOCATABLE:: left_snd_buffer(:),right_snd_buffer(:),bottom_snd_buffer(:),top_snd_buffer(:)

     TYPE(field_type):: field

  END TYPE chunk_type


  TYPE(chunk_type),  ALLOCATABLE       :: chunks(:)
  INTEGER                              :: number_of_chunks

  TYPE(grid_type)                      :: grid

END MODULE definitions_module
