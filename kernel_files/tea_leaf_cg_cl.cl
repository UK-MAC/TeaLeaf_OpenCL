/*
 *  Kernels used for conjugate gradient method
 */

#define CONDUCTIVITY 1
#define RECIP_CONDUCTIVITY 2

__kernel void tea_leaf_cg_init_u
(__global const double * __restrict const density1,
 __global const double * __restrict const energy1,
 __global       double * __restrict const u,
 __global       double * __restrict const p,
 __global       double * __restrict const r,
 __global       double * __restrict const d,
 const int coefficient)
{
    __kernel_indexes;

    p[THARR2D(0, 0, 0)] = 0.0;
    r[THARR2D(0, 0, 0)] = 0.0;

    if (row >= (y_min + 1) - 2 && row <= (y_max + 1) + 2
    && column >= (x_min + 1) - 2 && column <= (x_max + 1) + 2)
    {
        u[THARR2D(0, 0, 0)] = energy1[THARR2D(0, 0, 0)]*density1[THARR2D(0, 0, 0)];

        if (CONDUCTIVITY == coefficient)
        {
            d[THARR2D(0, 0, 0)] = density1[THARR2D(0, 0, 0)];
        }
        else
        {
            d[THARR2D(0, 0, 0)] = 1.0/density1[THARR2D(0, 0, 0)];
        }
    }
}

__kernel void tea_leaf_cg_init_directions
(__global const double * __restrict const d,
 __global       double * __restrict const ae,
 __global       double * __restrict const an,
 __global       double * __restrict const aw,
 __global       double * __restrict const as)
{
    __kernel_indexes;

    if (row >= (y_min + 1) - 1 && row <= (y_max + 1) + 1
    && column >= (x_min + 1) - 1 && column <= (x_max + 1) + 1)
    {
        ae[THARR2D(0, 0, 0)] = (d[THARR2D(1, 0, 0)] + d[THARR2D(0, 0, 0)])
            /(2.0*d[THARR2D(1, 0, 0)]*d[THARR2D(0, 0, 0)]);
        aw[THARR2D(0, 0, 0)] = (d[THARR2D(-1, 0, 0)] + d[THARR2D(0, 0, 0)])
            /(2.0*d[THARR2D(-1, 0, 0)]*d[THARR2D(0, 0, 0)]);
        an[THARR2D(0, 0, 0)] = (d[THARR2D(0, 1, 0)] + d[THARR2D(0, 0, 0)])
            /(2.0*d[THARR2D(0, 1, 0)]*d[THARR2D(0, 0, 0)]);
        as[THARR2D(0, 0, 0)] = (d[THARR2D(0, -1, 0)] + d[THARR2D(0, 0, 0)])
            /(2.0*d[THARR2D(0, -1, 0)]*d[THARR2D(0, 0, 0)]);
    }
}

__kernel void tea_leaf_cg_init_others
(__global       double * __restrict const bb,
 __global       double * __restrict const rro,
 __global       double * __restrict const p,
 __global       double * __restrict const r,
 __global       double * __restrict const w,
 __global       double * __restrict const b,
 __global const double * __restrict const u,
 __global const double * __restrict const ae,
 __global const double * __restrict const an,
 __global const double * __restrict const aw,
 __global const double * __restrict const as,
 double rx, double ry,
 __global       double * __restrict const z)
{
    __kernel_indexes;

    __local double bb_shared[BLOCK_SZ];
    __local double rro_shared[BLOCK_SZ];

    bb_shared[lid] = 0.0;
    rro_shared[lid] = 0.0;

    w[THARR2D(0, 0, 0)] = 0.0;
    b[THARR2D(0, 0, 0)] = 0.0;
    z[THARR2D(0, 0, 0)] = 0.0;

    if (row >= (y_min + 1) - 1 && row <= (y_max + 1) + 1
    && column >= (x_min + 1) - 1 && column <= (x_max + 1) + 1)
    {
        w[THARR2D(0, 0, 0)] = (1.0
            + ry*(an[THARR2D(0, 0, 0)] + as[THARR2D(0, 0, 0)])
            + rx*(ae[THARR2D(0, 0, 0)] + aw[THARR2D(0, 0, 0)]))*u[THARR2D(0, 0, 0)]
            - ry*(an[THARR2D(0, 0, 0)]*u[THARR2D(0, 1, 0)] + as[THARR2D(0, 0, 0)]*u[THARR2D(0, -1, 0)])
            - rx*(ae[THARR2D(0, 0, 0)]*u[THARR2D(1, 0, 0)] + aw[THARR2D(0, 0, 0)]*u[THARR2D(-1, 0, 0)]);

        r[THARR2D(0, 0, 0)] = u[THARR2D(0, 0, 0)] - w[THARR2D(0, 0, 0)];

#ifdef CG_DO_PRECONDITION
        b[THARR2D(0, 0, 0)] = 1.0/(1.0
            + ry*(an[THARR2D(0, 0, 0)] + as[THARR2D(0, 0, 0)])
            + rx*(ae[THARR2D(0, 0, 0)] + aw[THARR2D(0, 0, 0)]));

        z[THARR2D(0, 0, 0)] = b[THARR2D(0, 0, 0)]*r[THARR2D(0, 0, 0)];
        p[THARR2D(0, 0, 0)] = z[THARR2D(0, 0, 0)];
        rro_shared[lid] = r[THARR2D(0, 0, 0)]*z[THARR2D(0, 0, 0)];
#else
        p[THARR2D(0, 0, 0)] = r[THARR2D(0, 0, 0)];
        rro_shared[lid] = r[THARR2D(0, 0, 0)]*r[THARR2D(0, 0, 0)];
#endif

        bb_shared[lid] = u[THARR2D(0, 0, 0)]*u[THARR2D(0, 0, 0)];
    }

    REDUCTION(bb_shared, bb, SUM)
    REDUCTION(rro_shared, rro, SUM)
}

/* reduce rro/bb */

__kernel void tea_leaf_cg_solve_calc_w
(__global       double * __restrict const pw,
 __global const double * __restrict const p,
 __global       double * __restrict const w,
 __global const double * __restrict const ae,
 __global const double * __restrict const an,
 __global const double * __restrict const aw,
 __global const double * __restrict const as,
 double rx, double ry)
{
    __kernel_indexes;

    __local double pw_shared[BLOCK_SZ];

    pw_shared[lid] = 0;

    if (row >= (y_min + 1) - 1 && row <= (y_max + 1) + 1
    && column >= (x_min + 1) - 1 && column <= (x_max + 1) + 1)
    {
        w[THARR2D(0, 0, 0)] = (1.0
            + ry*(an[THARR2D(0, 0, 0)] + as[THARR2D(0, 0, 0)])
            + rx*(ae[THARR2D(0, 0, 0)] + aw[THARR2D(0, 0, 0)]))*p[THARR2D(0, 0, 0)]
            - ry*(an[THARR2D(0, 0, 0)]*p[THARR2D(0, 1, 0)] + as[THARR2D(0, 0, 0)]*p[THARR2D(0, -1, 0)])
            - rx*(ae[THARR2D(0, 0, 0)]*p[THARR2D(1, 0, 0)] + aw[THARR2D(0, 0, 0)]*p[THARR2D(-1, 0, 0)]);
        
        pw_shared[lid] = p[THARR2D(0, 0, 0)]*w[THARR2D(0, 0, 0)];
    }

    REDUCTION(pw_shared, pw, SUM);
}

/* reduce pw */

__kernel void tea_leaf_cg_solve_calc_ur
(double rro,
 __global const double * __restrict const pw,
 __global       double * __restrict const rrn,
 __global const double * __restrict const p,
 __global       double * __restrict const r,
 __global const double * __restrict const w,
 __global       double * __restrict const u,
 __global       double * __restrict const z,
 __global const double * __restrict const b)
{
    __kernel_indexes;

    __local double rrn_shared[BLOCK_SZ];

    rrn_shared[lid] = 0;

    const double alpha = rro/pw[0];

    if (row >= (y_min + 1) - 1 && row <= (y_max + 1) + 1
    && column >= (x_min + 1) - 1 && column <= (x_max + 1) + 1)
    {
        u[THARR2D(0, 0, 0)] += alpha*p[THARR2D(0, 0, 0)];
        r[THARR2D(0, 0, 0)] -= alpha*w[THARR2D(0, 0, 0)];
#ifdef CG_DO_PRECONDITION
        z[THARR2D(0, 0, 0)] = b[THARR2D(0, 0, 0)]*r[THARR2D(0, 0, 0)];
        rrn_shared[lid] = r[THARR2D(0, 0, 0)]*z[THARR2D(0, 0, 0)];
#else
        rrn_shared[lid] = r[THARR2D(0, 0, 0)]*r[THARR2D(0, 0, 0)];
#endif
    }

    REDUCTION(rrn_shared, rrn, SUM);
}

/* reduce rrn */

__kernel void tea_leaf_cg_solve_calc_p
(double rro,
 __global const double * __restrict const rrn,
 __global       double * __restrict const p,
 __global const double * __restrict const r,
 __global const double * __restrict const u,
 __global const double * __restrict const z)
{
    __kernel_indexes;

    const double beta = rrn[0]/rro;

    if (row >= (y_min + 1) - 1 && row <= (y_max + 1) + 1
    && column >= (x_min + 1) - 1 && column <= (x_max + 1) + 1)
    {
#ifdef CG_DO_PRECONDITION
        p[THARR2D(0, 0, 0)] = z[THARR2D(0, 0, 0)] + beta*p[THARR2D(0, 0, 0)];
#else
        p[THARR2D(0, 0, 0)] = r[THARR2D(0, 0, 0)] + beta*p[THARR2D(0, 0, 0)];
#endif
    }
}

