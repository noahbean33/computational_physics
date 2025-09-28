/* Author    : Okan Koeksal
 Copyright   : Free to modify 
 This code has been written by Okan Koeksal in the course of the C++ tutorials
 If you have any questions or find bugs, feel free to mail me "koeksal.okan@gmail.com"*/

#include <iostream>
#include <vector>
#include <fstream>
#include <iomanip>

using namespace std;

//define the number of columns in the file
int const COLS = 3;

int main()
{
	//2D-array as a vector of vectors
	vector<vector<double> > data; 
	vector<double> rowVector (COLS);
	int row = 0;

	ifstream myFile;
	myFile.open("C:\\Temp\\data.txt", ios::in);

	if (myFile.is_open() )
	{
		cout << "Successfully opened the file!" << endl;

		while (!myFile.eof() )
		{
			data.push_back(rowVector);
			for (int col = 0; col < COLS; col++)
			{
				myFile >> data[row][col];		//data[0][0]...data[0][2]
			}
			row++;
		}
	}
	else 
	{
		cout << "Unable to open the file!" << endl;
		exit (1);
	}

	myFile.close();

	//print the double values in the file
	for (int i = 0; i < data.size(); i++)
	{
		for (int j = 0; j < data[i].size(); j++)
		{
			cout << fixed << setprecision(1) << data[i][j] << " ";
		}
		cout << endl;
	}
	cout << endl;

	//calculate the sum of each row
	vector<double> rowSum;
	for (int i = 0; i < data.size(); i++)
	{
		double rowSum = 0.0;
		for (int j = 0; j < data[i].size(); j++)
		{
			rowSum += data[i][j];				//data[0][0]...data[0][2]
		}
		cout << "The sum of row #" << i+1 << ": " << rowSum << endl;
	}
	cout << endl;

	//calculate the sum of each col
	vector<double> colSum;
	for (int i = 0; i < data.size(); i++)
	{
		double colSum = 0.0;
		for (int j = 0; j < data[i].size(); j++)
		{
			colSum += data[j][i];			//data[0][0]...data[2][0]
		}
		cout << "The sum of col #" << i+1 << ": " << colSum << endl;
	}


return 0;
}