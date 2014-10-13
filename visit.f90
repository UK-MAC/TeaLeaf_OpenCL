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

!>  @brief Generates graphics output files.
!>  @author David Beckingsale, Wayne Gaudin
!>  @details The field data over all mesh chunks is written to a .vtk files and
!>  the .visit file is written that defines the time for each set of vtk files.
!>  The ideal gas and viscosity routines are invoked to make sure this data is
!>  up to data with the current energy, density and velocity.

SUBROUTINE visit

  USE clover_module
  USE update_halo_module
  USE viscosity_module
  USE ideal_gas_module

  IMPLICIT NONE

  INTEGER :: j,k,c,err,get_unit,u,dummy
  INTEGER :: nxc,nyc,nxv,nyv,nblocks
  REAL(KIND=8)    :: temp_var

  CHARACTER(len=80)           :: name
  CHARACTER(len=10)           :: chunk_name,step_name
  CHARACTER(len=90)           :: filename

  LOGICAL, SAVE :: first_call=.TRUE.

  INTEGER :: fields(NUM_FIELDS)

  REAL(KIND=8) :: kernel_time,timer

  name = 'clover'

  IF(first_call) THEN

    nblocks=number_of_chunks
    filename = "tea.visit"
    u=get_unit(dummy)
    OPEN(UNIT=u,FILE=filename,STATUS='UNKNOWN',IOSTAT=err)
    WRITE(u,'(a,i5)')'!NBLOCKS ',nblocks
    CLOSE(u)

    first_call=.FALSE.

  ENDIF

  IF(profiler_on) kernel_time=timer()
  DO c=1,chunks_per_task
    CALL ideal_gas(c,.FALSE.)
  ENDDO
  IF(profiler_on) profiler%ideal_gas=profiler%ideal_gas+(timer()-kernel_time)

  fields=0
  fields(FIELD_PRESSURE)=1
  fields(FIELD_XVEL0)=1
  fields(FIELD_YVEL0)=1
  IF(profiler_on) kernel_time=timer()
  CALL update_halo(fields,1)
  IF(profiler_on) profiler%halo_exchange=profiler%halo_exchange+(timer()-kernel_time)

  IF(profiler_on) kernel_time=timer()
  CALL viscosity()
  IF(profiler_on) profiler%viscosity=profiler%viscosity+(timer()-kernel_time)

  IF ( parallel%boss ) THEN

    filename = "tea.visit"
    u=get_unit(dummy)
    OPEN(UNIT=u,FILE=filename,STATUS='UNKNOWN',POSITION='APPEND',IOSTAT=err)

    DO c = 1, chunks_per_task
      WRITE(chunk_name, '(i6)') c+100000
      chunk_name(1:1) = "."
      WRITE(step_name, '(i6)') step+100000
      step_name(1:1) = "."
      filename = trim(trim(name) //trim(chunk_name)//trim(step_name))//".vtk"
      WRITE(u,'(a)')TRIM(filename)
    ENDDO
    CLOSE(u)

  ENDIF

  IF(profiler_on) kernel_time=timer()
  DO c = 1, chunks_per_task
    IF(chunks(c)%task.EQ.parallel%task) THEN
      nxc=chunks(c)%field%x_max-chunks(c)%field%x_min+1
      nyc=chunks(c)%field%y_max-chunks(c)%field%y_min+1
      nxv=nxc+1
      nyv=nyc+1
      WRITE(chunk_name, '(i6)') c+100000
      chunk_name(1:1) = "."
      WRITE(step_name, '(i6)') step+100000
      step_name(1:1) = "."
      filename = trim(trim(name) //trim(chunk_name)//trim(step_name))//".vtk"
      u=get_unit(dummy)
      OPEN(UNIT=u,FILE=filename,STATUS='UNKNOWN',IOSTAT=err)
      WRITE(u,'(a)')'# vtk DataFile Version 3.0'
      WRITE(u,'(a)')'vtk output'
      WRITE(u,'(a)')'ASCII'
      WRITE(u,'(a)')'DATASET RECTILINEAR_GRID'
      WRITE(u,'(a,2i12,a)')'DIMENSIONS',nxv,nyv,' 1'
      WRITE(u,'(a,i5,a)')'X_COORDINATES ',nxv,' double'
      DO j=chunks(c)%field%x_min,chunks(c)%field%x_max+1
        WRITE(u,'(e12.4)')chunks(c)%field%vertexx(j)
      ENDDO
      WRITE(u,'(a,i5,a)')'Y_COORDINATES ',nyv,' double'
      DO k=chunks(c)%field%y_min,chunks(c)%field%y_max+1
        WRITE(u,'(e12.4)')chunks(c)%field%vertexy(k)
      ENDDO
      WRITE(u,'(a)')'Z_COORDINATES 1 double'
      WRITE(u,'(a)')'0'
      WRITE(u,'(a,i20)')'CELL_DATA ',nxc*nyc
      WRITE(u,'(a)')'FIELD FieldData 5'
      WRITE(u,'(a,i20,a)')'density 1 ',nxc*nyc,' double'
      DO k=chunks(c)%field%y_min,chunks(c)%field%y_max
        WRITE(u,'(e12.4)')(chunks(c)%field%density0(j,k),j=chunks(c)%field%x_min,chunks(c)%field%x_max)
      ENDDO
      WRITE(u,'(a,i20,a)')'energy 1 ',nxc*nyc,' double'
      DO k=chunks(c)%field%y_min,chunks(c)%field%y_max
        WRITE(u,'(e12.4)')(chunks(c)%field%energy0(j,k),j=chunks(c)%field%x_min,chunks(c)%field%x_max)
      ENDDO
      WRITE(u,'(a,i20,a)')'pressure 1 ',nxc*nyc,' double'
      DO k=chunks(c)%field%y_min,chunks(c)%field%y_max
        WRITE(u,'(e12.4)')(chunks(c)%field%pressure(j,k),j=chunks(c)%field%x_min,chunks(c)%field%x_max)
      ENDDO
      WRITE(u,'(a,i20,a)')'viscosity 1 ',nxc*nyc,' double'
      DO k=chunks(c)%field%y_min,chunks(c)%field%y_max
        DO j=chunks(c)%field%x_min,chunks(c)%field%x_max
          temp_var=0.0
          IF(chunks(c)%field%viscosity(j,k).GT.0.00000001) temp_var=chunks(c)%field%viscosity(j,k)
          WRITE(u,'(e12.4)') temp_var
        ENDDO
      ENDDO
      WRITE(u,'(a,i20,a)')'temperature 1 ',nxc*nyc,' double'
      DO k=chunks(c)%field%y_min,chunks(c)%field%y_max
        WRITE(u,'(e12.4)')(chunks(c)%field%u(j,k),j=chunks(c)%field%x_min,chunks(c)%field%x_max)
      ENDDO
      WRITE(u,'(a,i20)')'POINT_DATA ',nxv*nyv
      WRITE(u,'(a)')'FIELD FieldData 2'
      WRITE(u,'(a,i20,a)')'x_vel 1 ',nxv*nyv,' double'
      DO k=chunks(c)%field%y_min,chunks(c)%field%y_max+1
        DO j=chunks(c)%field%x_min,chunks(c)%field%x_max+1
          temp_var=0.0
          IF(ABS(chunks(c)%field%xvel0(j,k)).GT.0.00000001) temp_var=chunks(c)%field%xvel0(j,k)
          WRITE(u,'(e12.4)') temp_var
        ENDDO
      ENDDO
      WRITE(u,'(a,i20,a)')'y_vel 1 ',nxv*nyv,' double'
      DO k=chunks(c)%field%y_min,chunks(c)%field%y_max+1
        DO j=chunks(c)%field%x_min,chunks(c)%field%x_max+1
          temp_var=0.0
          IF(ABS(chunks(c)%field%yvel0(j,k)).GT.0.00000001) temp_var=chunks(c)%field%yvel0(j,k)
          WRITE(u,'(e12.4)') temp_var
        ENDDO
      ENDDO
      CLOSE(u)
    ENDIF
  ENDDO
  IF(profiler_on) profiler%visit=profiler%visit+(timer()-kernel_time)

END SUBROUTINE visit

  subroutine visitcommandcallback (cmd, lcmd, args, largs)
  implicit none
  character*8 cmd, args
  integer     lcmd, largs
  end

  integer function visitbroadcastintfunction(value, sender)
  implicit none
  integer value, sender
!     REPLACE WITH MPI COMMUNICATION IF SIMULATION IS PARALLEL
  visitbroadcastintfunction = 0
  end

  integer function visitbroadcaststringfunction(str, lstr, sender)
  implicit none
  character*8 str
  integer     lstr, sender
!     REPLACE WITH MPI COMMUNICATION IF SIMULATION IS PARALLEL
  visitbroadcaststringfunction = 0
  end

  subroutine visitslaveprocesscallback ()
  implicit none
!     REPLACE WITH MPI COMMUNICATION IF SIMULATION IS PARALLEL
  end

  integer function visitactivatetimestep()
  implicit none
  include "visitfortransimV2interface.inc"
  visitactivatetimestep = VISIT_OKAY
  end

  integer function visitgetmetadata()
  implicit none
  include "visitfortransimV2interface.inc"
      integer runflag, simcycle, m1
      real simtime
      common /SIMSTATE/ runflag, simcycle, simtime

      integer md, err

      if(visitmdsimalloc(md).eq.VISIT_OKAY) then
          err = visitmdsimsetcycletime(md, simcycle, simtime)
          if(runflag.eq.1) then
              err = visitmdsimsetmode(md, VISIT_SIMMODE_RUNNING)
          else
              err = visitmdsimsetmode(md, VISIT_SIMMODE_STOPPED)
          endif

          if(visitmdmeshalloc(m1).eq.VISIT_OKAY) then
              err = visitmdmeshsetname(m1, "energy", 6)
              err = visitmdmeshsetmeshtype(m1, VISIT_MESHTYPE_RECTILINEAR)
              err = visitmdmeshsettopologicaldim(m1, 2)
              err = visitmdmeshsetspatialdim(m1, 2)
              err = visitmdmeshsetxunits(m1, "cells", 2)
              err = visitmdmeshsetyunits(m1, "cells", 2)
              err = visitmdmeshsetxlabel(m1, "Width", 5)
              err = visitmdmeshsetylabel(m1, "Height", 6)
              err = visitmdmeshsetcellorigin(m1, 1)
              err = visitmdmeshsetnodeorigin(m1, 1)

              err = visitmdsimaddmesh(md, m1)
          endif
      endif
      visitgetmetadata = md
  end

  integer function visitgetmesh(domain, name, lname)
  implicit none
  character*8 name
  integer     domain, lname, h, err, x, y
  integer     xmax, ymax
      common /SIMSIZE/ xmax, ymax
  integer, dimension(xmax) :: xs, ys
  include "visitfortransimV2interface.inc" 
      h = VISIT_INVALID_HANDLE
  xs = 1
  ys = 1
      if(visitstrcmp(name, lname, "energy", 6).eq.0) then
          if(visitrectmeshalloc(h).eq.VISIT_OKAY) then
              err = visitvardataalloc(x)
              err = visitvardataalloc(y)
              err = visitvardatasetf(x,VISIT_OWNER_SIM,1,xmax, xs)
              err = visitvardatasetf(y,VISIT_OWNER_SIM,1,ymax, ys)

              err = visitrectmeshsetcoordsxy(h, x, y)
          endif
      endif
      visitgetmesh = h
  end

  integer function visitgetvariable(domain, name, lname)
  implicit none
  character*8 name
  integer     domain, lname
  include "visitfortransimV2interface.inc"
  visitgetvariable = VISIT_INVALID_HANDLE
  end

  integer function visitgetcurve(name, lname)
  use definitions_module
  implicit none
  character*8 name
  integer     lname, h, nvals, err
  integer     xmax, ymax
      common /SIMSIZE/ xmax, ymax
  include "visitfortransimV2interface.inc"
      h = VISIT_INVALID_HANDLE

      if(visitstrcmp(name, lname, "energy", 6).eq.0) then
          if(visitvardataalloc(h).eq.VISIT_OKAY) then
              nvals = (xmax-1) * (ymax-1)
              err = visitvardatasetf(h, VISIT_OWNER_SIM,1,nvals, &
                chunks(1)%field%u)
          endif
      endif

      visitgetcurve = h
  end

  integer function visitgetdomainlist(name, lname)
  implicit none
  character*8 name
  integer     lname
  include "visitfortransimV2interface.inc"
  visitgetdomainlist = VISIT_INVALID_HANDLE
  end

  integer function visitgetdomainbounds(name, lname)
  implicit none
  character*8 name
  integer     lname
  include "visitfortransimV2interface.inc"
  visitgetdomainbounds = VISIT_INVALID_HANDLE
  end

  integer function visitgetdomainnesting(name, lname)
  implicit none
  character*8 name
  integer     lname
  include "visitfortransimV2interface.inc"
  visitgetdomainnesting = VISIT_INVALID_HANDLE
  end

  integer function visitgetmaterial(domain, name, lname)
  implicit none
  character*8 name
  integer     domain, lname
  include "visitfortransimV2interface.inc"
  visitgetmaterial = VISIT_INVALID_HANDLE
  end

