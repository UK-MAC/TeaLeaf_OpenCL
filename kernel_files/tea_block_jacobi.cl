
#define COEF_A (1*(-Ky[THARR2D(0,k+ 0, 0)]))
#define COEF_B (1*(1.0 + (Ky[THARR2D(0,k+ 1, 0)] + Ky[THARR2D(0,k+ 0, 0)]) + (Kx[THARR2D(1,k+ 0, 0)] + Kx[THARR2D(0,k+ 0, 0)])))
#define COEF_C (1*(-Ky[THARR2D(0,k+ 1, 0)]))

void block_solve_func
(__private const double r_l[JACOBI_BLOCK_SIZE],
 __global       double * __restrict const z,
 __global const double * __restrict const cp,
 __global const double * __restrict const bfp,
 __global const double * __restrict const Kx,
 __global const double * __restrict const Ky)
{
    const size_t column = get_global_id(0);
    const size_t row = get_global_id(1)*JACOBI_BLOCK_SIZE + 2;

    int k = 0;

    __private double dp_l[JACOBI_BLOCK_SIZE];
    __private double z_l[JACOBI_BLOCK_SIZE];

    dp_l[k] = r_l[k]/COEF_B;

    for (k = 1; k < BLOCK_TOP; k++)
    {
        dp_l[k] = (r_l[k] - COEF_A*dp_l[k - 1])*bfp[THARR2D(0, k, 0)];
    }

    k = BLOCK_TOP - 1;

    z_l[k] = dp_l[k];

    for (k = BLOCK_TOP - 2; k >= 0; k--)
    {
        z_l[k] = dp_l[k] - cp[THARR2D(0, k, 0)]*z_l[k + 1];
    }

    for (k = 0; k < BLOCK_TOP; k++)
    {
        z[THARR2D(0, k, 0)] = z_l[k];
    }
}

__kernel void block_solve
(__global const double * __restrict const r,
 __global       double * __restrict const z,
 __global const double * __restrict const cp,
 __global const double * __restrict const bfp,
 __global const double * __restrict const Kx,
 __global const double * __restrict const Ky)
{
    const size_t column = get_global_id(0);
    const size_t row = get_global_id(1)*JACOBI_BLOCK_SIZE + 2;

    if (row > y_max || column > x_max) return;

    __private double r_l[JACOBI_BLOCK_SIZE];

    for (int k = 0; k < BLOCK_TOP; k++)
    {
        r_l[k] = r[THARR2D(0, k, 0)];
    }

    block_solve_func(r_l, z, cp, bfp, Kx, Ky);
}

__kernel void block_init
(__global const double * __restrict const r,
 __global const double * __restrict const z,
 __global       double * __restrict const cp,
 __global       double * __restrict const bfp,
 __global const double * __restrict const Kx,
 __global const double * __restrict const Ky)
{
    const size_t column = get_global_id(0);
    const size_t row = get_global_id(1)*JACOBI_BLOCK_SIZE + 2;

    if (row > y_max || column > x_max) return;

    int k = 0;

    cp[THARR2D(0, k, 0)] = COEF_C/COEF_B;

    for (k = 1; k < BLOCK_TOP; k++)
    {
        bfp[THARR2D(0, k, 0)] = 1.0/(COEF_B - COEF_A*cp[THARR2D(0, k - 1, 0)]);
        cp[THARR2D(0, k, 0)] = COEF_C*bfp[THARR2D(0, k, 0)];
    }
}

