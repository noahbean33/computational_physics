/* Author    : Okan Koeksal
 Copyright   : Free to modify 
 This code has been written by Okan Koeksal in the course of the C++ tutorials
 If you have any questions or find bugs, feel free to mail me "koeksal.okan@gmail.com"*/

#include <iostream>

using namespace std;


const int CAPACITY = 10;

int* enlargeArray(int [], int CAPACITY);

int main()
{
	//initialize the first part of the array
	int array[] = {1,2,3,4,5,6,7,8,9,10};

	int* arrayPointer = enlargeArray(array, CAPACITY);

	for (int i = 0; i < CAPACITY * 2; i++)
	{
		cout << arrayPointer[i] << " ";
	}
	cout << endl;

	return 0;
}

int* enlargeArray(int array[], int CAPACITY) 
{
	//new pointer points to an array with 4 elements
	int* enlargedArray = new int [CAPACITY * 2];
	int value = 0;
	bool enteredNumberIn = false;

	do {
		for (int i = 0; i < CAPACITY*2; i++)
		{
			if (i < CAPACITY)
			{
				enlargedArray[i] = array[i];
			}
			else 
			{
				cout << "Please fill in the second half of the array by entering 10 numbers greater than 10: ";
				cin >> value;

				//ask for input and validate it
				while (value <= 10 || value == array[i])
				{
					enteredNumberIn = true;
					cout << "Invalid input, please fill in the second half of the array by typing in 10 numbers greater than 10: ";
					cin >> value;
				}
				enteredNumberIn = false;
				if (!enteredNumberIn) 
				{
					enlargedArray[i] = value;
				}

			}

		}


	}while(enteredNumberIn); //keep looping while "enteredNumberIn" is true and stays false
	return enlargedArray;
}
