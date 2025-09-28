/* Author    : Okan Koeksal
 Copyright   : Free to modify 
 This code has been written by Okan Koeksal in the course of the C++ tutorials
 If you have any questions or find bugs, feel free to mail me "koeksal.okan@gmail.com"*/

#include <iostream>

using namespace std; 

const int NUM_ELEMENTS = 6; 

void bubbleSort(int, int[], bool (*)(int, int));
bool ascendingSort(int, int); 
bool descendingSort(int, int);

int main() {

	int array[NUM_ELEMENTS];

	for (int i = 0; i < NUM_ELEMENTS; i++)
	{
		array[i] = (17*i+5) % 10;
	}
	cout << "Current array filled with arbitrary numbers: " << endl;

	for (int i = 0; i < NUM_ELEMENTS; i++)
	{
		cout << array[i] << " ";
	}
	cout << endl;

	bubbleSort(NUM_ELEMENTS, array, ascendingSort);
	cout << "Sorted in an ascending way: " << endl;

	for (int i = 0; i < NUM_ELEMENTS; i++)
	{
		cout << array[i] << " ";
	}
	cout << endl;

	bubbleSort(NUM_ELEMENTS, array, descendingSort);
	cout << "Sorted in an descending way: " << endl;

	for (int i = 0; i < NUM_ELEMENTS; i++)
	{
		cout << array[i] << " ";
	}
	cout << endl;


return 0; 
}

void bubbleSort(int numOfElements, int array[], bool (*compareNumbers)(int, int)) 
{
	bool wasElementSorted;
	do {
		wasElementSorted = true;
		//example: 5 2 9 6 3 0
		for (int i = 0; i < numOfElements -1; i++) 
		{
			if( (*compareNumbers)(array[i], array[i+1]) )
			{
				int temp = array[i]; 
				array[i] = array[i+1];
				array[i+1] = temp;
				wasElementSorted = false; 
			}

		}
	}while(!wasElementSorted);


}

bool ascendingSort(int x, int y) 
{
	if (x > y)
	{
		return true;
	}
	return false;
}

bool descendingSort(int x, int y) 
{
	if (x < y)
	{
		return true;
	}
	return false;
}