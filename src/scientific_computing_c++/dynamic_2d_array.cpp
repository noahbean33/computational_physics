/* Author    : Okan Koeksal
 Copyright   : Free to modify 
 This code has been written by Okan Koeksal in the course of the C++ tutorials
 If you have any questions or find bugs, feel free to mail me "koeksal.okan@gmail.com"*/

#include <iostream>

using namespace std;



int main() {
    
	int row, col;

	//declare array as a pointer to pointers
	int** array;

	//ask for user input
	cout << "Enter number of rows: ";
	cin >> row;
	cout << "Enter number of cols: ";
	cin >> col;

	//creation of an array of int pointers
	array = new int*[row];

	//whatever the value for row/col will be, so many times the array is created
	//e.g. if row, col = 2 then we will get a 2x2-matrix (4 integers)
	for (int i = 0; i < row; i++)
	{
		array[i] = new int[col];
	}

	for (int i = 0; i < row; i++)
	{
		for (int j = 0; j < col; j++)
		{
			//initialize all elements to zero
			//array[i][j] = 0;
			array[0][0] = 0;
			array[0][1] = 1;
			array[1][0] = 2;
			array[1][1] = 3;

		}

	}

		//memory deallocation: deleting the dynamically created array 
	for (int i = 0; i < row; i++)
	{
		delete [] array[i];
		delete [] array;

		//delete also pointer itself from memory
		array = NULL;
	}

	//print the matrix
	for (int i = 0; i < row; i++)
	{
		for (int j = 0; j < col; j++)
		{
			cout << array[i][j] << " ";
		}
		cout << endl;
	}

	//memory deallocation: deleting the dynamically created array 
	for (int i = 0; i < row; i++)
	{
		delete [] array[i];
		delete [] array;

		//delete also pointer itself from memory
		array = NULL;
	}



	return 0;
}