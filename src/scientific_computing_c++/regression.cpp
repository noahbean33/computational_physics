/* Author    : Okan Koeksal
 Copyright   : Free to modify 
 This code has been written by Okan Koeksal in the course of the C++ tutorials
 If you have any questions or find bugs, feel free to mail me "koeksal.okan@gmail.com"*/
 
#include "regression.h"

using namespace std;


int main()
{
	regression linear;
	linear.input();

	if (linear.coefficients())
	{
		linear.output();
		linear.correlation();
	}
	else 
		cout << "A best fit straight line is not available!" << endl;
		cout << endl;
	expoRegression exponential;
	exponential.input();

	if (exponential.coefficients())
	{
		exponential.output();
		exponential.correlation();
	}
	else 
		cout << "A best fit exponential curve is not available!" << endl;
		cout << endl;	

return 0;
}