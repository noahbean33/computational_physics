/* Author    : Okan Koeksal
 Copyright   : Free to modify 
 This code has been written by Okan Koeksal in the course of the C++ tutorials
 If you have any questions or find bugs, feel free to mail me "koeksal.okan@gmail.com"*/
 
#include <iostream>
#include <vector>


using namespace std; 

void fillVector (vector<int>&);
void printVector (const vector<int>&);
void sortVector (vector<int>&);
bool binarySearchAlgorithm (int, const vector<int>&);


int main () 
{
	vector<int> myVector;
	fillVector (myVector);
	printVector (myVector);
	sortVector (myVector);
	
	int enteredNumber = 0; 
	cout << "Please type in an int to compare with the numbers in the list: ";
	cin >> enteredNumber;

	if (binarySearchAlgorithm (enteredNumber, myVector) == true) 

		cout << "The number already exists! "<< endl;

	else 
		cout << "The number was not typed in before! " << endl;


	return 0;
}

void fillVector (vector<int>& newMyVector) 
{
	int input;
	cout << "Please type in the numbers and stop with (-1): ";
	cin >> input;

	while (input != -1) 
	{
		newMyVector.push_back(input);
		cin >> input;
	}
}

void printVector (const vector<int>& newMyVector) 
{
	cout << "The typed in numbers are: ";
	for (int i = 0; i < newMyVector.size(); i++) 
	{
		cout << newMyVector[i] << " ";
	}
	cout << endl;
}

void sortVector (vector<int>& newMyVector)
{
	int minI, minE = 0;
	for (int i = 0; i < newMyVector.size() -1; i++) 
	{
		minE = newMyVector[i];
		minI = i;
		for (int j = i + 1; j < newMyVector.size(); j++) 
		{
			if (newMyVector[j] < minE)
			{
				minE = newMyVector[j];
				minI = j;
			}
		}

			//2 4 6 1 3 5 --> in the first iteration "newMyVector[0] holds the value 2"
			int temp = newMyVector[i]; 
			//newMyVector[0] element (starting position) is assigned to the actual minimum value "1" being in the third index "newMyVector[3]"
			newMyVector[i] = minE;	
			//swapping the index of "1" where the minimum element was found with the corresponding index of the value of "2" 
			newMyVector[minI] = temp; 
	}
	printVector(newMyVector);
}

bool binarySearchAlgorithm (int enteredNumber, const vector<int>& newMyVector)
{
	int low = 0;
	int mid = 0;
	int high = newMyVector.size() - 1;

	while (low <= high) 
	{
		mid = (low + high) / 2;					//2 4 6 1 3 5 --> 1 2 3 4 5 6
		if (newMyVector[mid] == enteredNumber)	//(0 + 5) / 2 = 2
												//if number equals to newMyVector[mid], then return true
		{
			return true;
		}
		else if (newMyVector[mid] > enteredNumber) //if element at index 2 (value: 3) is greater than the searched number,
												   //move to the left by subtracting -1
		{
			high = mid - 1;
		}
		else									   //if none of above conditions are met, then the searched number has to be
												   //greater, move to the right by adding +1
			low = mid + 1;
	}
	return false;
}