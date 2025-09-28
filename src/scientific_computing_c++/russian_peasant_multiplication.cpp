/* Author    : Okan Koeksal
 Copyright   : Free to modify 
 This code has been written by Okan Koeksal in the course of the C++ tutorials
 If you have any questions or find bugs, feel free to mail me "koeksal.okan@gmail.com"*/

#include <iostream>
#include <cassert> 

using namespace std; 


int main() {
	
	int i, j;
	cout << "Please enter two non-negative numbers: ";
	cin >> i >> j; 
	//Precondition: i >= 0 && j >= 0 
	int x = i, y = j, z = 0; 
	assert(z + x*y == i*j); //Assertion
	
	while (x > 0)	//Example: 2 1
	{
		if (x % 2 == 1) 
		{
			z += y;
		}
		y *= 2;		//First cycle: x = i = 2; y = j = 1 --> (0 + 2*1 == 2*1)
		x /= 2;		//Second cycle: z = 2, y = 4, x = 0 --> (2 + 0*4 == 2*1)
		assert(z + x*y == i*j); //While loop invariant 

	}
	assert(z + x*y == i*j); //Postcondition 
	cout << "The product is: " << z << endl; 


return 0; 
}

	//int x, y;
	//cout << "Please enter two non-negative numbers: ";
	//cin >> x >> y;

	//int i = 0; 
	//while (x > 0)
	//{
	//	int j = 0; 
	//	while (j < y) //Example: 2 2
	//	{
	//		i++;
	//		j++;
	//	}
	//	x--;
	//}
	//cout << "The product is: " << i << endl;