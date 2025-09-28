//Parent class "Nonlinear"

#include <iostream>
#include <cfloat> 

#ifndef NONLINEAR_H
#define NONLINEAR_H

typedef double(*funct)(double);

class Nonlinear 
{
public:
funct Function; 
Nonlinear (funct F) { Function = F; }
//numerical derivation
double fprime (double);
//virtual function serving as a template for the Child class
virtual double iteration() = 0;
};

double Nonlinear::fprime(double x)
{
    //step size n is assigned to "FLT_EPSILON" (1e-5)
    double n = FLT_EPSILON;

    //keep the step size small, therefore we'll multiply and assign the step size to the absolute value of x
    //
    if (fabs(x) > 1)
	{
        n *= fabs(x);
	}
    //centered difference quotient
    return (Function(x + n) - Function(x - n))/ (2 * n);
}

#endif