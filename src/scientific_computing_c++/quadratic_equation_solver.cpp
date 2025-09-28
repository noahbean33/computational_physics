/* Author    : Okan Koeksal
 Copyright   : Free to modify 
 This code has been written by Okan Koeksal in the course of the C++ tutorials
 If you have any questions or find bugs, feel free to mail me "koeksal.okan@gmail.com"*/

#include <iostream>
#include <complex>
#include <exception>

using namespace std; 

typedef complex<double> complexNumber;

void quadraticEquation(double, double, double, complexNumber&, complexNumber&);

int main()
{
	double a, b, c;
	complexNumber x1, x2;
	cout << "Solution to the quadratic equation: a*x*x+b*x+c=0\n";
	cout << "Enter the coefficients a, b and c: ";
	cin >> a >> b >> c;

	quadraticEquation(a, b, c, x1, x2);
	cout << "x1 = " << x1 << endl;
	cout << "x2 = " << x2 << endl;

	return 0; 
}

void quadraticEquation(double a, double b, double c, complexNumber& x1, complexNumber& x2)
{
	double discriminant;
	// The complex numbers are entered as (real, imag)
	x1 = complexNumber(0,0);
	x2 = complexNumber(0,0);

	try{
		discriminant = b*b - 4.0*a*c;
		if(fabs(a) < 1e-14)
		{
			throw runtime_error("No quadratic equation was entered!\n");
		}

		if(discriminant >= 0)
		{
			double d = sqrt(discriminant);
			x1 = complexNumber( (-b+d)/(2*a), 0);
			x2 = complexNumber( (-b-d)/(2*a), 0);
		}

		else
		{
			double d = sqrt(fabs(discriminant));
			x1 = complexNumber( -b/(2*a), d/(2*a) );
			x2 = complexNumber( -b/(2*a), -d/(2*a) );
		}
	}

	catch(const exception& error)
	{
		cerr << error.what();
		exit(1);
	}
}
