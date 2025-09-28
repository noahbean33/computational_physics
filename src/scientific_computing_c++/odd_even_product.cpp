/* Author    : Okan Koeksal
 Copyright   : Free to modify 
 This code has been written by Okan Koeksal in the course of the C++ tutorials
 If you have any questions or find bugs, feel free to mail me "koeksal.okan@gmail.com"*/

#include <iostream>

using namespace std; 

//prototypes
int evenNumbers (int);
int oddNumbers (int);

int main () 
{
	//Alternative solution without functions
	int productOfOddNumbers = 1;
	int productOfEvenNumbers = 1;
	int counter = 0;

	for (int counter = 1; counter <= 7; counter++)
	{
		if (counter % 2 != 0)
		{
		productOfOddNumbers *= counter;
		}
		else  
		{
		productOfEvenNumbers *= counter;
		}

	}
	
	cout << "The product of even numbers is: " << productOfEvenNumbers << endl;
	cout << "The product of odd numbers is: " << productOfOddNumbers << endl;
		
	/*int numbers = 0;
	cout << "The product of even numbers is: " << evenNumbers(numbers) << endl;
	cout << "The product of odd numbers is: " << oddNumbers(numbers) << endl;*/
	
	return 0;
}

/*
int evenNumbers (int number)
{
	int productOfEvenNumbers = 1;

	for (int counter = 1; counter <= 7; counter++)
	{
		if (counter % 2 == 0)
		{
		productOfEvenNumbers *= counter;   //1*2*4*6 = 48
		}
	}
	return productOfEvenNumbers;
}


int oddNumbers (int number)
{
	int productOfOddNumbers = 1;

	for (int counter = 1; counter <= 7; counter++)
	{
		if (counter % 2 != 0)
		{
		productOfOddNumbers *= counter;   //1*3*5*7 = 105
		}
	}
	return productOfOddNumbers;
}
*/