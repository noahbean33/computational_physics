/* Author    : Okan Koeksal
 Copyright   : Free to modify 
 This code has been written by Okan Koeksal in the course of the C++ tutorials
 If you have any questions or find bugs, feel free to mail me "koeksal.okan@gmail.com"*/
 
#include <iostream>
#include <cassert>

using namespace std; 

//constants: pi and p
const double pi = 3.141592, p = 0.231641;

class NormDistribution {
private:
	double x, mue, sigma;
public:
	//user-defined Constructor(parametrized)
	NormDistribution(double X, double M = 0, double S = 1) : x(X), mue(M), sigma(S)
	{
	cout << "Random variable: " << x << endl
		 << "Mean value: " << mue << endl
		 << "Standard deviation: " << sigma << endl;
	}

	//calculation of the z-score
	double probability();
	//approximation technique for solving the normal distribution by Hastings
	double stdNormDistribution(double z);
	//static member variable
	static double c[6];
};

//coefficients: 0.319381, -0.356563, 1.781477, -1.821255, 1.330274
double NormDistribution::c[6] = {0.0, 0.319381, -0.356563, 1.781477, -1.821255, 1.330274};
double NormDistribution::probability() { return stdNormDistribution( (x - mue) / sigma); }

double NormDistribution::stdNormDistribution(double z)
{
	//assure that sigma > 0
	assert(sigma > 0);
	double k = 1.0 / (1.0 + p * fabs(z));
	//Density function
	double d = exp(-z*z / 2.0) / sqrt(2.0 * pi);
	//initial value used in our for-loop
	double f = c[5];

	//Start of Horner's method: evaluate the value of the polynomial using Horner's method
	//factored form of polynomial function: P(x) ~ 1 - d(x) * k*(c_1 + k*(c_2 + k*(c_3 + k*(c_4 + c_5 * k)))) + c_0 
	for (int i = 4; i >= 0; i--)
	{
		f = f*k + c[i];
	}
	//last step is to multiply the result obtained from the for-loop by the density function
	f *= d;

	//conditional operator for the z-score
	return (z < 0)? f:1.0 -f;

	//Alternative way
	/*if (z < 0)
	{
		return z = f;
	}
	else 
		return z = 1.0 -f;*/
}


int main()
{
	NormDistribution personsInfluenza(985., 1000., 12.);
	cout << 1. - personsInfluenza.probability() 
		<< " is the probability that at least one person among 985 persons is affected by influenza!" << endl;
	cout << personsInfluenza.probability() 
		<< " is the probability that at least one person among 985 persons is not affected by influenza!" << endl;

return 0;
}