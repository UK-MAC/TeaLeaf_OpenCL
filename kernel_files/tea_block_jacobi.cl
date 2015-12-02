
#define COEF_A (-Ky[THARR2D(0,k+ 0, 0)])
#define COEF_B (1.0 + (Ky[THARR2D(0,k+ 1, 0)] + Ky[THARR2D(0,k+ 0, 0)]) + (Kx[THARR2D(1,k+ 0, 0)] + Kx[THARR2D(0,k+ 0, 0)]))
#define COEF_C (-Ky[THARR2D(0,k+ 1, 0)])

void block_solve_func
(kernel_info_t kernel_info,
 __SHARED__ const double r_local[BLOCK_SZ],
 __SHARED__       double z_local[BLOCK_SZ],
 __GLOBAL__ const double * __restrict const cp,
 __GLOBAL__ const double * __restrict const bfp,
 __GLOBAL__ const double * __restrict const Kx,
 __GLOBAL__ const double * __restrict const Ky)
{
    const int column = get_global_id(0);
    const int row = get_global_id(1);

    const int loc_column = get_local_id(0);
    const int loc_row_size = LOCAL_X;

    const int x_max = kernel_info.x_max;
    const int y_max = kernel_info.y_max;

    const int upper_limit = BLOCK_TOP;

    int k = 0;
#define LOC_K (loc_column + k*loc_row_size)

    __private double dp_priv[JACOBI_BLOCK_SIZE];

    dp_priv[k] = r_local[LOC_K]/COEF_B;

    for (k = 1; k < upper_limit; k++)
    {
        dp_priv[k] = (r_local[LOC_K] - COEF_A*dp_priv[k - 1])*bfp[THARR2D(0, k, 0)];
    }

    k = upper_limit - 1;

    z_local[LOC_K] = dp_priv[k];

    for (k = upper_limit - 2; k >= 0; k--)
    {
        z_local[LOC_K] = dp_priv[k] - cp[THARR2D(0, k, 0)]*z_local[LOC_K + LOCAL_X];
    }
}

__kernel void tea_leaf_block_solve
(kernel_info_t kernel_info,
 __GLOBAL__ const double * __restrict const r,
 __GLOBAL__       double * __restrict const z,
 __GLOBAL__ const double * __restrict const cp,
 __GLOBAL__ const double * __restrict const bfp,
 __GLOBAL__ const double * __restrict const Kx,
 __GLOBAL__ const double * __restrict const Ky)
{
    __kernel_indexes;

    __SHARED__ double r_l[BLOCK_SZ];
    __SHARED__ double z_l[BLOCK_SZ];

    r_l[lid] = 0;
    z_l[lid] = 0;

    if (WITHIN_BOUNDS)
    {
        r_l[lid] = r[THARR2D(0, 0, 0)];
    }

    barrier(CLK_LOCAL_MEM_FENCE);
    if (loc_row == 0)
    {
        block_solve_func(kernel_info,r_l, z_l, cp, bfp, Kx, Ky);
    }
    barrier(CLK_LOCAL_MEM_FENCE);

    if (WITHIN_BOUNDS)
    {
        z[THARR2D(0, 0, 0)] = z_l[lid];
    }
}

__kernel void tea_leaf_block_init
(kernel_info_t kernel_info,
 __GLOBAL__ const double * __restrict const r,
 __GLOBAL__ const double * __restrict const z,
 __GLOBAL__       double * __restrict const cp,
 __GLOBAL__       double * __restrict const bfp,
 __GLOBAL__ const double * __restrict const Kx,
 __GLOBAL__ const double * __restrict const Ky)
{
    __kernel_indexes;

    const int upper_limit = BLOCK_TOP;

    if (WITHIN_BOUNDS)
    {
        if (loc_row == 0)
        {
            int k = 0;

            cp[THARR2D(0, k, 0)] = COEF_C/COEF_B;

            for (k = 1; k < upper_limit; k++)
            {
                bfp[THARR2D(0, k, 0)] = 1.0/(COEF_B - COEF_A*cp[THARR2D(0, k - 1, 0)]);
                cp[THARR2D(0, k, 0)] = COEF_C*bfp[THARR2D(0, k, 0)];
            }
        }
    }
}

