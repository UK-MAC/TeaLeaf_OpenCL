
__kernel void viscosity
(__global const double * __restrict const celldx,
 __global const double * __restrict const celldy,
 __global const double * __restrict const density0,
 __global const double * __restrict const pressure,
 __global       double * __restrict const viscosity,
 __global const double * __restrict const xvel0,
 __global const double * __restrict const yvel0)
{
    __kernel_indexes;

    double ugrad, vgrad, grad2, pgradx, pgrady, pgradx2, pgrady2,
        grad, ygrad, pgrad, xgrad, div, strain2, limiter;

    if(row >= (y_min + 1) && row <= (y_max + 1)
    && column >= (x_min + 1) && column <= (x_max + 1))
    {
        ugrad = (xvel0[THARR2D(1, 0, 1)] + xvel0[THARR2D(1, 1, 1)])
              - (xvel0[THARR2D(0, 0, 1)] + xvel0[THARR2D(0, 1, 1)]);

        vgrad = (yvel0[THARR2D(0, 1, 1)] + yvel0[THARR2D(1, 1, 1)])
              - (yvel0[THARR2D(0, 0, 1)] + yvel0[THARR2D(1, 0, 1)]);
        
        div = (celldx[column] * ugrad) + (celldy[row] * vgrad);

        strain2 = 0.5 * (xvel0[THARR2D(0, 1, 1)] + xvel0[THARR2D(1, 1, 1)]
                - xvel0[THARR2D(0, 0, 1)] - xvel0[THARR2D(1, 0, 1)])/celldy[row]
                + 0.5 * (yvel0[THARR2D(1, 0, 1)] + yvel0[THARR2D(1, 1, 1)]
                - yvel0[THARR2D(0, 0, 1)] - yvel0[THARR2D(0, 1, 1)])/celldx[column];

        pgradx = (pressure[THARR2D(1, 0, 0)] - pressure[THARR2D(-1, 0, 0)])
               / (celldx[column] + celldx[column + 1]);
        pgrady = (pressure[THARR2D(0, 1, 0)] - pressure[THARR2D(0, -1, 0)])
               / (celldy[row] + celldy[row + 1]);

        pgradx2 = pgradx*pgradx;
        pgrady2 = pgrady*pgrady;

        limiter = ((0.5 * ugrad / celldx[column]) * pgradx2
                + ((0.5 * vgrad / celldy[row]) * pgrady2)
                + (strain2 * pgradx * pgrady))
                / MAX(pgradx2 + pgrady2, 1.0e-16);


        pgradx = SIGN(MAX(1.0e-16, fabs(pgradx)), pgradx);
        pgrady = SIGN(MAX(1.0e-16, fabs(pgrady)), pgrady);
        pgrad = SQRT((pgradx * pgradx) + (pgrady * pgrady));

        xgrad = fabs(celldx[column] * pgrad / pgradx);
        ygrad = fabs(celldy[row] * pgrad / pgrady);

        grad = MIN(xgrad, ygrad);
        grad2 = grad * grad;

        if(limiter > 0 || div >= 0.0)
        {
            viscosity[THARR2D(0,0,0)] = 0.0;
        }
        else
        {
            viscosity[THARR2D(0,0,0)] = 2.0 * density0[THARR2D(0,0,0)] * grad2 * (limiter * limiter);
        }
    }
}

