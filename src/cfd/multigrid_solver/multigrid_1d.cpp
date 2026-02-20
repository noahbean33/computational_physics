// ONE DIMENSIONAL MULTIGRID CODE
// C++ equivalent of multigrid_1d.f90
// code is for educational purposes only
// solves d^2(phi)/dx^2=0
// solution fixed to zero on boundaries at i=1 and i=max
// possible grid points: 3,5,9,17,33,65,129,257,513,1025,etc.(these include boundary points)
// iterations over interior points from imin to imax
// lmax is number of grid levels (1 is no refinement, Gauss-Seidel solver)
// max is maximum number of grid points on fine grid
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
#include <chrono>
#include <iomanip>

// Type aliases (1-based: allocate size n+1, ignore index 0)
using Vec1D  = std::vector<double>;
using Vec2D  = std::vector<Vec1D>;
using IVec1D = std::vector<int>;

//***********************************************************************
void gauss_seidel(int itermax, int imin, int imax, const IVec1D& stride,
                  const Vec1D& h, int l, Vec2D& x, const Vec2D& b)
{
    for (int iter = 1; iter <= itermax; iter++)
        for (int i = imin; i <= imax; i += stride[l])
            x[i][l] = (x[i+stride[l]][l] + x[i-stride[l]][l]
                       - h[l]*h[l]*b[i][l]) / 2.0;
}

//***********************************************************************
//     compute error residuals and transfer down to coarser grids
void restriction(int imin, int imax, const IVec1D& stride,
                 const Vec1D& h, int l, const Vec2D& x, Vec2D& b)
{
    // note b(i,l+1) is the residual vector r for solving Ae=r on next coarse level
    // direct transfer on down to coarser grid; no interpolation necessary
    for (int i = imin; i <= imax; i += stride[l+1])
        b[i][l+1] = b[i][l] - (x[i+stride[l]][l] + x[i-stride[l]][l]
                    - 2.0*x[i][l]) / (h[l]*h[l]);
}

//***********************************************************************
//     interpolate errors up from coarser grids to finer grids
void prolongation(int imin, int imax, const IVec1D& stride, int l, Vec2D& x)
{
    // direct transfer of error up across grids
    for (int i = imin + stride[l]; i <= imax - stride[l]; i += 2*stride[l]) {
        double correction = x[i][l+1];
        x[i][l] += correction;
    }
    // linear interpolation of error between points moving to finer grid
    for (int i = imin; i <= imax; i += 2*stride[l]) {
        double correction = 0.5*(x[i+stride[l]][l+1] + x[i-stride[l]][l+1]);
        x[i][l] += correction;
    }
}

//***********************************************************************
int main()
{
    const double pi = 4.0 * std::atan(1.0);

    int max_pts, lmax, numouter, itermax = 1;

    // set some parameters
    std::cout << "Enter the number of grid points in one direction" << std::endl;
    std::cout << "i.e., 3,5,9,17,33,65,129,257,513,1025,2049,4097:  ";
    std::cin >> max_pts;
    std::cout << std::endl;

    std::cout << "Maximum number of multigrid levels is: "
              << static_cast<int>(std::round(
                     std::log(static_cast<double>(max_pts - 1)) / std::log(2.0)))
              << std::endl;
    std::cout << "Enter number of grid levels:  ";
    std::cin >> lmax;
    std::cout << std::endl;

    std::cout << "Enter the maximum number of multigrid cycles:  ";
    std::cin >> numouter;
    std::cout << std::endl;

    if (lmax > 1) {
        std::cout << "Enter the number of Gauss-Seidel iterations per level:  ";
        std::cin >> itermax;
        std::cout << std::endl;
    }

    // Allocate with 1-based indexing (size+1, index 0 unused)
    Vec2D  x(max_pts + 1, Vec1D(lmax + 1, 0.0));
    Vec2D  b(max_pts + 1, Vec1D(lmax + 1, 0.0));
    Vec1D  h(lmax + 1, 0.0);
    IVec1D stride(lmax + 1, 0);

    // Open output files
    std::ofstream sol_file, res_file, err_file;
    if (lmax == 1) {
        sol_file.open("solution.gs");
        res_file.open("residual.gs");
        err_file.open("error.gs");
    } else {
        sol_file.open("solution.mg");
        res_file.open("residual.mg");
        err_file.open("error.mg");
    }

    double ymax = 1.0;               // domain length
    // b is already 0.0               // right hand side of Ax=b
    h[1] = ymax / (max_pts - 1);     // grid spacing on finest mesh
    stride[1] = 1;                    // stride at one on finest mesh

    // compute coarser grid spacings and strides
    for (int i = 2; i <= lmax; i++) {
        h[i] = 2.0 * h[i-1];
        stride[i] = 2 * stride[i-1];
    }

    // set initial condition on dependent variable on fine grid
    for (int i = 1; i <= max_pts; i++)
        x[i][1] = std::sin(10.0 * (i - 1) * pi / (max_pts - 1.0));

    // (reset) boundary conditions
    for (int l = 1; l <= lmax; l++) {
        x[1][l] = 0.0;
        x[max_pts][l] = 0.0;
    }

    auto t_start = std::chrono::high_resolution_clock::now();

    int imin, imax;
    double rms;

    for (int outer = 1; outer <= numouter; outer++) {
        //*******************************************************************
        //     work down the cycle
        // initialize the error solutions to zero
        for (int l = 2; l <= lmax; l++)
            for (int i = 1; i <= max_pts; i++)
                x[i][l] = 0.0;

        for (int l = 1; l <= lmax; l++) {
            imin = stride[l] + 1;
            imax = max_pts - stride[l];
            gauss_seidel(itermax, imin, imax, stride, h, l, x, b);
            if (l == lmax) break;   // don't restrict to lmax+1 grid level
            if (lmax == 1) break;   // don't call restriction if no multigrid
            imin = stride[l+1] + 1;
            imax = max_pts - stride[l+1];
            restriction(imin, imax, stride, h, l, x, b);
        }

        //********************************************************************
        //     work up the cycle by interpolating errors to finer meshes
        for (int l = lmax - 1; l >= 1; l--) {
            imin = stride[l] + 1;
            imax = max_pts - stride[l];
            prolongation(imin, imax, stride, l, x);
            // smoothing iterations of error on way up after interpolation
            gauss_seidel(3, imin, imax, stride, h, l, x, b);
        }

        //**********************************************************************
        //     compute and write RMS residuals on finest grid
        rms = 0.0;
        for (int i = 2; i <= max_pts - 1; i++) {
            double residual = b[i][1] - (x[i+1][1] + x[i-1][1]
                              - 2.0*x[i][1]) / (h[1]*h[1]);
            rms += residual * residual;
        }
        rms = std::sqrt(rms);

        // MAXVAL(ABS(x(:,1)))
        double max_abs_x = 0.0;
        double max_x = 0.0;
        for (int i = 1; i <= max_pts; i++) {
            if (std::abs(x[i][1]) > max_abs_x) max_abs_x = std::abs(x[i][1]);
            if (x[i][1] > max_x) max_x = x[i][1];
        }

        std::cout << " After cycle" << std::setw(7) << outer
                  << "   RMS residual=" << std::scientific << std::setprecision(5) << rms
                  << "   maximum error=" << std::scientific << std::setprecision(5) << max_abs_x
                  << std::endl;

        res_file << outer << " " << rms << std::endl;
        err_file << outer << " " << max_x << std::endl;

        if (rms <= 1.0e-07) break;   // stop if converged
    }

    // write solution to file
    for (int i = 1; i <= max_pts; i++)
        sol_file << i << " " << x[i][1] << std::endl;

    auto t_end = std::chrono::high_resolution_clock::now();
    double walltime = std::chrono::duration<double>(t_end - t_start).count();

    std::cout << std::endl << std::endl;
    std::cout << "Elapsed time= " << walltime << std::endl << std::endl;
    std::cout << "Solution written to solution.xx" << std::endl;
    std::cout << "RMS residuals written to residual.xx" << std::endl;
    std::cout << "Solution errors written to error.xx" << std::endl;
    std::cout << std::endl;

    return 0;
}
//***********************************************************************
