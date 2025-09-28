/* Author    : Okan Koeksal
 Copyright   : Free to modify 
 This code has been written by Okan Koeksal in the course of the C++ tutorials
 If you have any questions or find bugs, feel free to mail me "koeksal.okan@gmail.com"*/

#include <iostream>
#include <ctime>

using namespace std;


const int NUM_ELEMENTS = 4; 
int array[NUM_ELEMENTS][NUM_ELEMENTS] = { {0,0,0,0}, {0,0,0,0}, {0,0,0,0}, {0,0,0,0}  }; 

void generateRandomNumbers(); 
void fillArray(int);
void printArray();

int main () {

	generateRandomNumbers();
	printArray();

	return 0;
}

void generateRandomNumbers()
{

	int randomNumbers = 0; 
	//array declaration and initialization of all elements to 0
	//int array[] = {0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0}; 

	//declaration of our pointer to a dynamic array of datatype int with a size equal to 16 integers 
	int* array = new int[16];
	for (int i = 0; i < 16; i++)
	{
		array[i] = 0;
	}

	//seed the generated numbers
	srand (time(0));
	bool numberToBePicked = true; 

	//iterate 16 times for 16 different numbers ranging from 1 to 16 
	for (int i = 0; i < NUM_ELEMENTS*NUM_ELEMENTS; i++)
	{

		do {
			numberToBePicked = true; 
			randomNumbers = 1 + rand() % 17;

			if (array[randomNumbers-1] == 0)
			{
				array[randomNumbers-1] = randomNumbers;
				numberToBePicked = false;
			}
		}while(numberToBePicked);

	fillArray(randomNumbers);
	}

	//deletes the allocated block of memory 
	delete []array;
}

void fillArray(int randomNumber)
{
	for (int i = 0; i < NUM_ELEMENTS; i++)
	{
		for (int j = 0; j < NUM_ELEMENTS; j++)
		{
			if (array[i][j] == 0)
			{
				array[i][j] = randomNumber;
				return;
			}

		}

	}

}

void printArray()
{
	for (int i = 0; i < NUM_ELEMENTS; i++)
	{
		for (int j = 0; j < NUM_ELEMENTS; j++)
		{
			cout << array[i][j] << " ";
		}
		cout << endl;
	}

}
