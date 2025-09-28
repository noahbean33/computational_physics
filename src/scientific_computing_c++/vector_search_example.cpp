/* Author    : Okan Koeksal
 Copyright   : Free to modify 
 This code has been written by Okan Koeksal in the course of the C++ tutorials
 If you have any questions or find bugs, feel free to mail me "koeksal.okan@gmail.com"*/

#include <iostream>
#include <vector>

using namespace std;

//filling the vector
void fillVector (vector<int>&);

//printing the vector
void printVector (const vector<int>&);

//boolean-function for checking whether the number already exists or not
bool searchVector (int, vector<int>&);


int main () {

	vector<int> myVector;
	fillVector(myVector);
	printVector(myVector);

	int enteredNumber = 0;
	cout << "Please type in an int to compare with the numbers in the list: ";
	cin >> enteredNumber;

	if (searchVector(enteredNumber, myVector) == true ) 
	{
		cout << "The number already exists!" << endl;
	}

	else 
		cout << "The number was not typed in before!" << endl;

return 0;
}


void fillVector(vector<int>& newMyVector) 
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

void printVector(const vector<int>& newMyVector) 
{
	cout << "The existing numbers are: ";
	for (int i = 0; i < newMyVector.size(); i++) 
	{
		cout << newMyVector[i] << " ";
	}
	cout << endl;
}


bool searchVector (int enteredNumber, vector<int>& newMyVector) 
{	
	for (int i = 0; i < newMyVector.size(); i++) 
	{
		if (enteredNumber == newMyVector[i]) 
		{
			return true;
		}
	}
		return false;
}