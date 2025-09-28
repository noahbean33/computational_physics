/* Author    : Okan Koeksal
 Copyright   : Free to modify 
 This code has been written by Okan Koeksal in the course of the C++ tutorials
 If you have any questions or find bugs, feel free to mail me "koeksal.okan@gmail.com"*/

#include <iostream>

using namespace std; 

typedef double(*function)(double);
double function_1(double);
double function_2(double);
function arrayFunctionPointer[] = {function_1, function_2};
void tabularValue(function, double, double, double);

int main()
{
	//For the first called function: starting at i = 0 and is incremented up to i = 1
	for (int i = 0; i < 2; i++)
	{
		tabularValue(arrayFunctionPointer[i], i, i+1, 0.2);
		cout << endl;
	}

return 0; 
}

double function_1(double x) 
{
	return sqrt(x);
}

double function_2(double x) 
{
	return exp(x);
}


void tabularValue(function f, double a, double b, double n)
{
	//For the first iteration of the first called function: x = a = 0, b = 1 --> 1 < 1.1
	//For the first iteration of the second called function: x = a = 1, b = 2 --> 2 < 2.1
	for (double x = a; x < b + n/2; x += n)
	{
		cout << x << "\t" << f(x) << endl;
	}

}
