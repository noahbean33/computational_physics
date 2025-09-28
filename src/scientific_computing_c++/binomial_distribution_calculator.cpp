/* Author    : Okan Koeksal
 Copyright   : Free to modify 
 This code has been written by Okan Koeksal in the course of the C++ tutorials
 If you have any questions or find bugs, feel free to mail me "koeksal.okan@gmail.com"*/
 
#include <iostream>
#include <iomanip>
#include <cassert>

using namespace std;

class Binomial 
{
private:
	int n, k;
	double p;	
public:
	//user-defined constructor (parametrized constructor)
	Binomial (int N, int K, double P) {n = N; k = K; p = P;}
	//single probability
	double singleValue();
	//binomial table of all probabilities
	void binomialTable();
	//expected value
	double expectedValue();
};

double Binomial::singleValue()
{
	double b, r;
	double q = 1.0 - p;
	int m = n + 1;
	//assert()-funct. for different conditions
	assert(p > 0 && p < 1 && k > 0 && k <= n);
	//trick: work with logs in order to avoid numerical underflow
	//log(q^n) = n * log(q)
	b = n * log(q);
	r = log(p/q);
	for (int i = 1; i <= k; i++)
	{
		//n - k + 1 / k = n + 1 / k - k/k --> n + 1 / k - 1
		b += r + log(static_cast<double>(m) / i - 1.0);
	}
	//exp(ln(x)) = x
	return exp(b);
}

void Binomial::binomialTable()
{
	double b, r;
	double q = 1.0 - p;
	int m = n + 1;
	//assert()-funct. for different conditions
	assert(p > 0 && p < 1);
	//trick: work with logs in order to avoid numerical underflow
	//log(q^n) = n * log(q)
	b = n * log(q);
	r = log(p/q);

	cout << fixed << setprecision(6) << setw(5) << "0" << setw(15) << exp(b) << endl;
	for (int i = 1; i <= n; i++)
	{
		//n - k + 1 / k = n + 1 / k - k/k --> n + 1 / k - 1
		b += r + log(static_cast<double>(m) / i - 1.0);
		cout << setw(5) << i << setw(15) << exp(b) << endl;
	}
}

double Binomial::expectedValue()
{
	return n * p;
}


int main()
{
	//B(n,k,p)
	Binomial postedAnswer(6,1,0.7);
	cout << "Probability p for one right answer is: " << postedAnswer.singleValue() << endl;
	cout << "Expected value: " << postedAnswer.expectedValue() << endl;
	cout << "Table of binomial probabilities: " << endl;
	postedAnswer.binomialTable();

	Binomial tossingCoin(4,1,0.5);
	cout << "Probability p for one head is: " << tossingCoin.singleValue() << endl;
	cout << "Expected value: " << tossingCoin.expectedValue() << endl;
	cout << "Table of binomial probabilities: " << endl;
	tossingCoin.binomialTable();

	return 0;
}