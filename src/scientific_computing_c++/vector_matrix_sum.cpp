/* Author    : Okan Koeksal
 Copyright   : Free to modify 
 This code has been written by Okan Koeksal in the course of the C++ tutorials
 If you have any questions or find bugs, feel free to mail me "koeksal.okan@gmail.com"*/

#include <iostream>
#include <vector>

using namespace std;

void defineVector (vector<vector<int> >&);
void printVector (const vector<vector<int> >&);
int calculateSum (vector<vector<int> >&);

int main()
{
	vector<vector<int> > matrix;
	defineVector (matrix);
	printVector (matrix);
	cout << "The sum is: " << calculateSum(matrix) << endl;

	
	return 0;
}

void defineVector (vector<vector<int> >& myVector) 
{
	int row, col; 
	cout << "Please type in the number of rows: ";
	cin >> row;
	cout << "Please type in the number of cols: ";
	cin >> col; 

	for (int i = 0; i < row; i++)
	{
		//temporary vector for i, j
		vector<int> temp;
		for (int j = 0; j < col; j++)
		{
			//add j to the temporary vector
			temp.push_back(j);
		}
		//push the temporary vector back onto the vector of vectors 
		myVector.push_back(temp);
	}

	int input;
	cout << "Please type in the exact amount of numbers matching your matrix dimension (" << row*col << " numbers): ";  

	for (int i = 0; i < myVector.size(); i++)
	{
		for (int j = 0; j < myVector[i].size(); j++)
		{
			cin >> input;
			myVector[i][j] = input;
		}

	}

}

void printVector (const vector<vector<int> >& myVector) 
{
	cout << "Vector of vector is filled with numbers: ";
	for (int i = 0; i < myVector.size(); i++) 
	{
		for (int j = 0; j < myVector[i].size(); j++)
		{
			cout << myVector[i][j] << " ";
		}

	}
	cout << endl;

}

int calculateSum (vector<vector<int> >& myVector)
{
	int total = 0; 
	for (int i = 0; i < myVector.size(); i++)
	{
		for (int j = 0; j < myVector[i].size(); j++)
		{
			total += myVector[i][j];
		}

	}
	return total;
}
