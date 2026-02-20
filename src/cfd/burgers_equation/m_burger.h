#ifndef M_BURGER_H
#define M_BURGER_H

///@file m_burger.h
///@brief C++ equivalent of m_burger.f90
///@details Module containing FWD, TLM and ADJ for 1D Burger's equations.
/// Original author: Seon Ki Park (03/24/98). Ported from Fortran-90 to C++.

#include <vector>

namespace m_burger {

// Constants
constexpr double R      = 100.0;                    // Reynolds number (reciprocal of diffusion coefficient)
constexpr double dx     = 1.0;                      // space increment
constexpr double dt     = 0.1;                      // time increment
constexpr double dtdx   = dt / dx;
constexpr double dtdxsq = dt / (dx * dx);
constexpr double c1     = (2.0 / R) * dtdxsq;
constexpr double c0     = 1.0 / (1.0 + c1);

// Array types: Vec2D is indexed as u[i][j] where i=spatial, j=temporal (0-based)
using Vec1D = std::vector<double>;
using Vec2D = std::vector<std::vector<double>>;

/// Forward model of 1D Burger's equations.
/// @param nx    Number of grid points
/// @param n     Number of time steps
/// @param ui    Initial conditions (length nx)
/// @param uob   Observations (nx x n); overwritten with model solution on exit
/// @param u     Model solutions (nx x n)
/// @param cost  Cost function (output)
void burger(int nx, int n, Vec1D& ui, Vec2D& uob, Vec2D& u, double& cost);

/// Tangent Linear Model of the Burger's equations.
/// @param nx     Number of grid points
/// @param n      Number of time steps
/// @param ui     Initial conditions (length nx); overwritten with u(:,n) on exit
/// @param ubasic Basic state (nx x n)
/// @param u      TLM solutions (nx x n)
void burger_tlm(int nx, int n, Vec1D& ui, const Vec2D& ubasic, Vec2D& u);

/// Adjoint model of the Burger's equations.
/// @param iforcing  Forcing flag (0 or 1)
/// @param nx        Number of grid points
/// @param n         Number of time steps
/// @param ui        Initial/adjoint variable (length nx)
/// @param ubasic    Basic state (nx x n)
/// @param uob       Observations (nx x n)
/// @param u         Adjoint solutions (nx x n)
void burger_adj(int iforcing, int nx, int n, Vec1D& ui, Vec2D& ubasic, Vec2D& uob, Vec2D& u);

} // namespace m_burger

#endif // M_BURGER_H
