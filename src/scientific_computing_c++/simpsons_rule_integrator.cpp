/* Author    : Okan Koeksal
 Copyright   : Free to modify 
 This code has been written by Okan Koeksal in the course of the C++ tutorials
 If you have any questions or find bugs, feel free to mail me "koeksal.okan@gmail.com"*/

#include <iostream>
#include <cassert>
#include <iomanip>

const  double pi = 3.141592;
const int n = 22; 

using namespace std; 

typedef double(*function)(double);
double usualIntegral(double);
double exponentialIntegral(double);
double simpsonsRule(double, double, function);


int main() {

	assert(n % 2 == 0);
	cout << "Solution to the first integral: " << fixed << setprecision(6) << simpsonsRule(0,3,usualIntegral) << endl;
	cout << "Solution to the second integral: " << fixed << setprecision(6) << simpsonsRule(0.1,1.3,exponentialIntegral) << endl;


return 0; 
}

double usualIntegral(double x)
{
	if (fabs(x) < DBL_EPSILON)
	{
		return 1.0;
	}
	else
		return 1/(1+pow(x,5));
}

double exponentialIntegral(double x)
{
	if (fabs(x) < DBL_EPSILON)
	{
		return 1.0;
	}
	else 
		return 5*x*exp(-2*x);
}

double simpsonsRule(double a, double b, function f)
{
	double coefficient = 2.0;
	double delta_x = (b - a) / n; // 3 - 0 / 2 = 3/2
	double simpson = f(a) + f(b);

	for (int i = 1; i < n; i++)
	{
		simpson += (coefficient = 6 - coefficient)* f(a + i*delta_x);
	}
	return simpson* delta_x / 3; // (3/2)/3 * [1*1/(1+0^5) + 4*1/(1+(3/2)^5) + 1*1/(1+3^5)]
}
