// TWO DIMENSIONAL MULTIGRID CODE
// C++ equivalent of multigrid_2d.f90
// code is for educational purposes only
// two dimensional poisson equation using multigrid solver
// solves del^2 (phi)=-100; phi=zero on boundaries
// domain :   -1<=x<=1  and -1<=y<=1
// solution fixed to 0 on boundaries at i=1 and i=max; j=1 and j=jmax
// iterations over interior points from imin to imax and jmin to jmax
// centerline temperature series solution is 29.4685...
// possible grid points: 3,5,9,17,33,65,129,257,513 etc. in each direction
// level l=1 represents the finest mesh
// stride is a multiple of grid spacing; i.e., stride(l)=2*stride(l-1)
//
// to run pure Gauss-Seidel w/o multigrid, specify 1 grid level and number of multigrid cycles
//                                    equal to the number of Gauss-Seidel iterations desired.
//
// Note: arrays use 1-based indexing (index 0 is unused) to preserve the
//       Fortran stencil arithmetic exactly.

#include <iostream>
#include <fstream>
#include <vector>
#include <cmath>
#include <iomanip>

// Type aliases (1-based: allocate size n+1, ignore index 0)
using Vec1D  = std::vector<double>;
using Vec2D  = std::vector<Vec1D>;
using Vec3D  = std::vector<Vec2D>;
using IVec1D = std::vector<int>;

//***************************************************************************************
void gauss_seidel(int itermax, int imin, int imax, int jmin, int jmax,
                  const IVec1D& stride, const Vec1D& h, int l,
                  Vec3D& x, const Vec3D& b)
{
    for (int iter = 1; iter <= itermax; iter++)
        for (int j = jmin; j <= jmax; j += stride[l])
            for (int i = imin; i <= imax; i += stride[l])
                x[i][j][l] = 0.25*(x[i+stride[l]][j][l] + x[i-stride[l]][j][l]
                            + x[i][j+stride[l]][l] + x[i][j-stride[l]][l])
                            - h[l]*h[l]*b[i][j][l] / 4.0;
}

//***************************************************************************************
//     compute error residuals and transfer down to coarser grids
void restriction(int imin, int imax, int jmin, int jmax,
                 const IVec1D& stride, const Vec1D& h, int l,
                 const Vec3D& x, Vec3D& b)
{
    for (int i = imin; i <= imax; i += stride[l])
        for (int j = jmin; j <= jmax; j += stride[l])
            b[i][j][l+1] = b[i][j][l]
                - (x[i+stride[l]][j][l] + x[i-stride[l]][j][l]
                 + x[i][j+stride[l]][l] + x[i][j-stride[l]][l]
                 - 4.0*x[i][j][l]) / (h[l]*h[l]);
}

//***************************************************************************************
//     interpolate solutions from coarser to finer grids
void prolongation(int imin, int imax, int jmin, int jmax,
                  const IVec1D& stride, int l, Vec3D& x)
{
    int s = stride[l];

    // 4 ave (circles on grid sketch)
    for (int j = jmin; j <= jmax; j += 2*s)
        for (int i = imin; i <= imax; i += 2*s) {
            double correction = 0.25*(x[i-s][j-s][l+1] + x[i+s][j+s][l+1]
                              + x[i-s][j+s][l+1] + x[i+s][j-s][l+1]);
            x[i][j][l] += correction;
        }

    // 2 ave (vertical average; triangles on grid sketch)
    for (int j = jmin; j <= jmax; j += 2*s)
        for (int i = imin + s; i <= imax - s; i += 2*s) {
            double correction = 0.5*(x[i][j+s][l+1] + x[i][j-s][l+1]);
            x[i][j][l] += correction;
        }

    // 2 ave (horizontal average; squares on grid sketch)
    for (int j = jmin + s; j <= jmax - s; j += 2*s)
        for (int i = imin; i <= imax; i += 2*s) {
            double correction = 0.5*(x[i+s][j][l+1] + x[i-s][j][l+1]);
            x[i][j][l] += correction;
        }

    // 0 ave (no interpolation; diamonds on grid sketch)
    for (int j = jmin + s; j <= jmax - s; j += 2*s)
        for (int i = imin + s; i <= imax - s; i += 2*s) {
            double correction = x[i][j][l+1];
            x[i][j][l] += correction;
        }
}

//***************************************************************************************
int main()
{
    int max_pts, lmax, numouter, itermax = 1;

    std::cout << "Enter the number of grid points in one direction" << std::endl;
    std::cout << " i.e., 3,5,9,17,33,65,129,257,513,1025,2049:  ";
    std::cin >> max_pts;
    std::cout << std::endl;

    std::cout << " Maximum number of multigrid levels is "
              << static_cast<int>(std::round(
                     std::log(static_cast<double>(max_pts - 1)) / std::log(2.0)))
              << std::endl;
    std::cout << " Enter number of grid levels:  ";
    std::cin >> lmax;
    std::cout << std::endl;

    std::cout << " Enter the maximum number of multigrid cycles:  ";
    std::cin >> numouter;
    std::cout << std::endl;

    if (lmax > 1) {
        std::cout << " Enter the number of Gauss-Seidel iterations per level:  ";
        std::cin >> itermax;
        std::cout << std::endl;
    }

    // Allocate with 1-based indexing (size+1, index 0 unused)
    Vec3D  x(max_pts + 1, Vec2D(max_pts + 1, Vec1D(lmax + 1, 0.0)));
    Vec3D  b(max_pts + 1, Vec2D(max_pts + 1, Vec1D(lmax + 1, -100.0)));
    Vec1D  h(lmax + 1, 0.0);
    IVec1D stride(lmax + 1, 0);

    // initialize: b=-100 (already done above), x=0 (already done above)
    h[1] = 2.0 / static_cast<double>(max_pts - 1);
    stride[1] = 1;

    for (int i = 2; i <= lmax; i++) {
        h[i] = 2.0 * h[i-1];
        stride[i] = 2 * stride[i-1];
    }

    int imin, imax, jmin, jmax;
    double rms;

    //*****************************************************************
    for (int outer = 1; outer <= numouter; outer++) {
        // initialize error solutions to zero
        for (int l = 2; l <= lmax; l++)
            for (int i = 1; i <= max_pts; i++)
                for (int j = 1; j <= max_pts; j++)
                    x[i][j][l] = 0.0;

        //*****************************************************************
        //     work down the cycle (smoothing or restriction)
        for (int l = 1; l <= lmax; l++) {
            imin = stride[l] + 1;  imax = max_pts - stride[l];
            jmin = stride[l] + 1;  jmax = max_pts - stride[l];
            gauss_seidel(itermax, imin, imax, jmin, jmax, stride, h, l, x, b);
            if (l == lmax) break;   // don't restrict to (nonexistent) lmax+1 grid level
            if (lmax == 1) break;   // don't call restriction if no multigrid
            imin = stride[l+1] + 1;  imax = max_pts - stride[l+1];
            jmin = stride[l+1] + 1;  jmax = max_pts - stride[l+1];
            restriction(imin, imax, jmin, jmax, stride, h, l, x, b);
        }

        //******************************************************************
        //     work up the cycle  (interpolation or prolongation)
        for (int l = lmax - 1; l >= 1; l--) {
            imin = stride[l] + 1;  imax = max_pts - stride[l];
            jmin = stride[l] + 1;  jmax = max_pts - stride[l];
            prolongation(imin, imax, jmin, jmax, stride, l, x);
            // smoothing iterations of error on way up after interpolation
            gauss_seidel(5, imin, imax, jmin, jmax, stride, h, l, x, b);
        }

        //******************************************************************
        // compute top level l=1 rms residual
        rms = 0.0;
        for (int i = 2; i <= max_pts - 1; i++)
            for (int j = 2; j <= max_pts - 1; j++) {
                double residual = b[i][j][1]
                    - (x[i+1][j][1] + x[i-1][j][1]
                     + x[i][j+1][1] + x[i][j-1][1]
                     - 4.0*x[i][j][1]) / (h[1]*h[1]);
                rms += residual * residual;
            }
        rms = std::sqrt(rms);

        int center = (max_pts + 1) / 2;
        std::cout << " After cycle" << std::setw(5) << outer
                  << "   RMS residual=" << std::scientific << std::setprecision(5) << rms
                  << "   center point result=" << std::scientific
                  << std::setprecision(5) << x[center][center][1]
                  << std::endl;

        if (rms <= 1.0e-05) break;   // exit loop if converged
    }

    // write solution to file for gnuplot contour
    std::cout << "Writing data file ..." << std::endl;
    std::ofstream data_file("results.dat");
    for (int i = 1; i <= max_pts; i++) {
        data_file << std::endl;
        for (int j = 1; j <= max_pts; j++)
            data_file << i << " " << j << " " << x[i][j][1] << std::endl;
    }
    std::cout << "Finished" << std::endl;

    return 0;
}
//***************************************************************************************
