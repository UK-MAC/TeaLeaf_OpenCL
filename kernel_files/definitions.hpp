#ifndef __CL_DEFINITIONS_HDR
#define __CL_DEFINITIONS_HDR

#define JACOBI_BLOCK_SIZE 4

#define DEFAULT_LOCAL_SIZE 128

// FIXME Change these to make more sense
#define DBGOUT stderr
#define LOCAL_Y (JACOBI_BLOCK_SIZE)
#define LOCAL_X (DEFAULT_LOCAL_SIZE/LOCAL_Y)

// used in update_halo and for copying back to host for mpi transfers
#define FIELD_density       1
#define FIELD_energy0       2
#define FIELD_energy1       3
#define FIELD_u             4
#define FIELD_p             5
#define FIELD_sd            6
#define FIELD_r             7
#define FIELD_z             8
#define NUM_FIELDS          8
#define FIELD_vector_p FIELD_p
#define FIELD_vector_sd FIELD_sd
#define FIELD_vector_r FIELD_r
#define FIELD_vector_z FIELD_z

#define NUM_BUFFERED_FIELDS 8

// which side to pack - keep the same as in fortran file
#define CHUNK_LEFT 1
#define CHUNK_left 1
#define CHUNK_RIGHT 2
#define CHUNK_right 2
#define CHUNK_BOTTOM 3
#define CHUNK_bottom 3
#define CHUNK_TOP 4
#define CHUNK_top 4
#define EXTERNAL_FACE       (-1)

#define CELL_DATA   1
#define VERTEX_DATA 2
#define X_FACE_DATA 3
#define Y_FACE_DATA 4

// preconditioners
#define TL_PREC_NONE        1
#define TL_PREC_JAC_DIAG    2
#define TL_PREC_JAC_BLOCK   3

#define TEA_ENUM_JACOBI     1
#define TEA_ENUM_CG         2
#define TEA_ENUM_CHEBYSHEV  3
#define TEA_ENUM_PPCG       4
#define TEA_ENUM_DPCG       5

// same as in fortran
#define COEF_CONDUCTIVITY 1
#define COEF_RECIP_CONDUCTIVITY 2

#endif
