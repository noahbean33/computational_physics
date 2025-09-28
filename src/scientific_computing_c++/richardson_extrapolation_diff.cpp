/* Author    : Okan Koeksal
 Copyright   : Free to modify 
 This code has been written by Okan Koeksal in the course of the C++ tutorials 
 If you have any questions or find bugs, feel free to mail me "koeksal.okan@gmail.com"*/

#include <iostream>
#include <iomanip>

using namespace std; 

//threshold value for the convergence criterion
const double tolerance = 1e-6; //0.00001
typedef double(*funct)(double);

//trigonometric functions
double function_1(double x) { return sin(x); }
double function_2(double x) { return cos(x); }

class numericalDifferentiation 
{
private:
	//point where the derivative is evaluated
	double x;
	//step size
	double n;
	funct Function;

public:
	numericalDifferentiation(double X, funct f) { x = X; Function = f; n = 1.0/2; }
	double differenceQuotient(void);
	void extrapolation(void);

};

double numericalDifferentiation::differenceQuotient(void)
{
	//centered difference quotient
	return (Function(x+n) - Function(x-n)) / (2.0*n);
}

void numericalDifferentiation::extrapolation(void) 
{
	int i, r = 0;
	//array for holding 10 elements (extrapolated values)
	double prefactor, D[10];
	for (i = 0; i <= 9; i++)
		//initialization of the array
		D[i] = 0.0;

	do {

		if (r >= 1)
		{
			//e.g. r = 4 --> i = 4 -1 = 3 ...
			for (int i = r - 1; i >= 0; i--)
			{
				//We set D[0] to D[1] ...
				D[i+1] = D[i];
			}

		}
		D[0] = differenceQuotient();
		prefactor = 1.0;
		cout << setw(2) << setprecision(6) << D[0];

		for (i = 1; i <= r; i++)
		{
			prefactor *= 4.0;
			D[i] = (prefactor * D[i-1] - D[i]) / (prefactor - 1.0);
			cout << setw(10) << setprecision(6) << D[i];
		}
		//step value
		n /= 2.0;
		r++;
		cout << endl;

	}while(r <= 1 || fabs(D[r-1] - D[r-2]) > tolerance*fabs(D[r-1]));
	cout << endl;

}



int main()
{
	numericalDifferentiation differentiate_f1(1.0, function_1);
	numericalDifferentiation differentiate_f2(1.0, function_2);
	differentiate_f1.extrapolation();
	differentiate_f2.extrapolation();

return 0; 
}