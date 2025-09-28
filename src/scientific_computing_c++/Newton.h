//Child class "Newton"

#include "Nonlinear.h"

using namespace std;

#ifndef NEWTON_H
#define NEWTON_H

class Newton : public Nonlinear
{
private:
    //initial guess (starting point)
    double xStart;
    //max. iteration steps
    int maxIterations;
    //accuracy
    double epsilon;

public:
    Newton(funct F, double X, int I, double E) :
		Nonlinear(F), xStart(X), maxIterations(I), epsilon(E) {}
    virtual double iteration();
};

double Newton::iteration()
{
    double denominator, y, x = xStart;
    int iteration = 0;

    while(iteration < maxIterations)
	{
        //assign the variable "denominator" to the central difference quotient as a function of x
        denominator = fprime(x);
        if (fabs(denominator) < DBL_EPSILON)
            cout << "Denominator is zero\n ";
        //Newton's interpolation formula
        y = x - Function(x) / denominator;
        if (fabs(y) > FLT_MAX)
        cout << "Overflow\n ";
        if (fabs (x-y) < epsilon*fabs(x))
            break;
        cout << "y = " << y << "\t" << "x = " << x << endl;
        iteration++;
        //assign x to y for using this value in the next iteration
        x = y;
	}
    if (iteration >= maxIterations)
	{
        x = 0;
        cout << "Maximum number of iteration steps are reached" << endl;
	}
    return y;
}

#endif 
