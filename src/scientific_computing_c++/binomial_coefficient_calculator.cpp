/* Author    : Okan Koeksal
 Copyright   : Free to modify 
 This code has been written by Okan Koeksal in the course of the C++ tutorials
 If you have any questions or find bugs, feel free to mail me "koeksal.okan@gmail.com"*/
 
#include <iostream>
#include <cassert> 

using namespace std;

//recursive function
/*long binomialCoefficient(int n, int k)
{
	//assert()-funct. to assure that non-physical numbers are not entered
	assert(n >= k && k >= 0);
	if (k == 0 || k == n)
	{
		return 1L;
	}
	else
		return (binomialCoefficient(n - 1, k - 1) + binomialCoefficient(n - 1, k));

}*/

/*
C(3,2) = 3! / 2! (3-2)! = 3
C(3,2) = Sum of
	= C(3-1,2-1) + (3-1,2)
	= C(2,1) + (2,2) (= 1)
	C(2,1) = Sum of
		C(1,1) = 1
		C(1,0) = 1
--> Sum of all these numbers: 3
*/

long binomialCoefficient(int n, int k)
{
	long binomial = 1L;
	assert(n >= k && k >= 0);
	if (k > n/2)
	{
		//symmetry rule
		k = n - k;
	}
	for (int i = 1; i <= k; i++)
	{
		//multiplicative formula
		//k = n - k = 3 - 2 = 1
		//1L * (3 - 1 + 1) / 1 = 3L
		binomial = binomial*(n - k + i) / i;
	}
	return binomial;
}


int main()
{
	int n, k; 
	cout << "Calculation of the binomial coefficient of n choose k\n";
	cout << "Please enter n and k: ";
	cin >> n >> k;
	cout << "The result of k out of n = " << binomialCoefficient(n,k) << " combinations" << endl;

	return 0;
}