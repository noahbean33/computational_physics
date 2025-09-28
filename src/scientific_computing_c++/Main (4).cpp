/* Author    : Okan Koeksal
 Copyright   : Free to modify 
 This code has been written by Okan Koeksal in the course of the C++ tutorials
 If you have any questions or find bugs, feel free to mail me "koeksal.okan@gmail.com"*/
 
#include "Newton.h"

double function_1(double x)
{
	return -(x)*(x)*(x) - 4*(x) + 10;
}

double function_2(double x)
{
	//alternatively use pow(x,3)
	return pow(x,3) - 2*(x) + 2;
}


int main()
{
	cout << "Calculation of root using the iterative method by Newton" << endl;
	Newton polynomial (function_1, 1, 10, 1e-6);
	Newton polynomial_2 (function_2, -1, 10, 1e-6);

	cout << "The approximated root for this function is: " << polynomial.iteration() << endl << endl;
	cout << "The approximated root for this function is: " << polynomial_2.iteration() << endl << endl;

return 0;
}