///@file m_burger.cpp
///@brief C++ equivalent of m_burger.f90
///@details Module containing FWD, TLM and ADJ for 1D Burger's equations.
/// Original author: Seon Ki Park (03/24/98). Ported from Fortran-90 to C++.
///
/// Note on array layout:
///   Fortran uses 1-based indexing; this C++ port uses 0-based indexing.
///   2D arrays are represented as Vec2D (vector of vectors) indexed as
///   u[i][j], where i is the spatial index (0..nx-1) and j is the time
///   index (0..n-1), matching the Fortran layout u(i,j).

#include "m_burger.h"

namespace m_burger {

void burger(int nx, int n, Vec1D& ui, Vec2D& uob, Vec2D& u, double& cost) {
    // initialize the cost function:
    cost = 0.0;

    // set the initial conditions:
    for (int i = 0; i < nx; i++)
        u[i][0] = ui[i];

    // set the boundary conditions:
    for (int j = 0; j < n; j++) {
        u[0][j] = 0.0;
        u[nx-1][j] = 0.0;
    }

    // integrate the model numerically:
    // FTCS for the 1st time step integration
    for (int i = 1; i < nx-1; i++)
        u[i][1] = u[i][0] - 0.5*dtdx*u[i][0]*(u[i+1][0] - u[i-1][0])
                 + 0.5*c1*(u[i+1][0] - 2.0*u[i][0] + u[i-1][0]);

    // Leapfrog/DuFort-Frankel afterwards
    for (int j = 2; j < n; j++)
        for (int i = 1; i < nx-1; i++)
            u[i][j] = c0*(u[i][j-2] + c1*(u[i+1][j-1] - u[i][j-2] + u[i-1][j-1]))
                     - dtdx*u[i][j-1]*(u[i+1][j-1] - u[i-1][j-1]);

    // cost function:
    for (int j = 0; j < n; j++)
        for (int i = 0; i < nx; i++)
            cost += 0.5 * (u[i][j] - uob[i][j]) * (u[i][j] - uob[i][j]);

    // save nonlinear solutions to the basic fields:
    for (int j = 0; j < n; j++)
        for (int i = 0; i < nx; i++)
            uob[i][j] = u[i][j];
}

void burger_tlm(int nx, int n, Vec1D& ui, const Vec2D& ubasic, Vec2D& u) {
    // set the initial conditions:
    for (int i = 0; i < nx; i++)
        u[i][0] = ui[i];

    // set the boundary conditions:
    for (int j = 0; j < n; j++) {
        u[0][j] = 0.0;
        u[nx-1][j] = 0.0;
    }

    // FTCS for 1st time step integration
    for (int i = 1; i < nx-1; i++)
        u[i][1] = u[i][0] - 0.5*dtdx*(ubasic[i][0]*(u[i+1][0] - u[i-1][0])
                 + u[i][0]*(ubasic[i+1][0] - ubasic[i-1][0]))
                 + 0.5*c1*(u[i+1][0] - 2.0*u[i][0] + u[i-1][0]);

    // Leapfrog/DuFort-Frankel afterwards
    for (int j = 2; j < n; j++)
        for (int i = 1; i < nx-1; i++)
            u[i][j] = c0*(u[i][j-2] + c1*(u[i+1][j-1] - u[i][j-2] + u[i-1][j-1]))
                     - dtdx*(u[i][j-1]*(ubasic[i+1][j-1] - ubasic[i-1][j-1])
                     + ubasic[i][j-1]*(u[i+1][j-1] - u[i-1][j-1]));

    // set the final value of u in ui:
    for (int i = 0; i < nx; i++)
        ui[i] = u[i][n-1];
}

void burger_adj(int iforcing, int nx, int n, Vec1D& ui, Vec2D& ubasic, Vec2D& uob, Vec2D& u) {
    // initialize adjoint variables:
    for (int j = 0; j < n; j++)
        for (int i = 0; i < nx; i++)
            u[i][j] = 0.0;

    // set the final conditions:
    if (iforcing == 0) {
        for (int i = 0; i < nx; i++) {
            u[i][n-1] = ui[i];
            ui[i] = 0.0;
        }
    } else {
        for (int i = 0; i < nx; i++) {
            u[i][n-1] = ubasic[i][n-1] - uob[i][n-1];
            ui[i] = 0.0;
        }
    }

    // adjoint of Leapfrog/DuFort-Frankel
    for (int j = n-1; j >= 2; j--)
        for (int i = nx-2; i >= 1; i--)
            u[i][j-2] += c0*(c1*(u[i+1][j-1] - u[i-1][j-1])
                       - dtdx*ubasic[i][j-1]*(u[i+1][j-1] - u[i-1][j-1]))
                       + ubasic[i][j-1]*(u[i+1][j-1] - 2.0*u[i][j-1] + u[i-1][j-1]);

    if (iforcing == 1) {
        for (int i = 0; i < nx; i++)
            u[i][0] += ubasic[i][0] - uob[i][0];
    }

    // adjoint of FTCS
    for (int i = nx-2; i >= 1; i--) {
        u[i-1][0] += 0.5*(c1 + dtdx*ubasic[i][0])*u[i][1];
        u[i][0]   += (1.0 - c1)*u[i][1];
        u[i+1][0] += 0.5*(c1 - dtdx*ubasic[i][0])*u[i][1];
    }

    if (iforcing == 1) {
        for (int i = 0; i < nx; i++)
            ui[i] += ubasic[i][0] - uob[i][0];
    }

    // set the boundary conditions:
    for (int j = 0; j < n; j++) {
        u[0][j] = 0.0;
        u[nx-1][j] = 0.0;
    }

    // set the final value of u in ui:
    for (int i = 0; i < nx; i++)
        ui[i] += u[i][0];
}

} // namespace m_burger
