/* Author    : Okan Koeksal
 Copyright   : Free to modify 
 This code has been written by Okan Koeksal in the course of the C++ tutorials
 If you have any questions or find bugs, feel free to mail me "koeksal.okan@gmail.com"*/
 
#include <iostream>
#include <string>
#include <iomanip>

using namespace std;

const int CAPACITY = 100;

int main () {

	//declaration of column-arrays 
	double col_1[CAPACITY];
	double col_2[CAPACITY];

	//initialization of all elements in these two arrays to zeros
	for (int i = 0; i < CAPACITY; i++)
	{
		col_1[i] = 0;
		col_2[i] = 0;
	}

	//declaration of linspace-array holding 10 elements
	double linspace[10];

	for (int i = 0; i < 10; i++)
	{
		linspace[i] = 0;
	}

	//linspace-array producing 10 elements ranging from 0.0 to 1.0
	double start, end = 0.0;
	//adjust the ranges
	start = 0.0;
	end = 1.0;

	for (int i = 0; i < 10; i++) 
	{
		double I = static_cast<double>(i);
		double n = static_cast<double>(10)-1;
		//x_1 + ( (x_2 - x_1) / n) * i
		linspace[i] = start + ((end - start) / n) * I;
		//cout << "Values produced by the linspace-function: " << linspace[i] << endl;
	}

	//distribute linspace-function in column 1
	for (int i = 0; i < 10; i++)
	{
		for (int j = 0; j < 10; j++)
		{
			//iteration scheme: i, j = 0 yield col_1[0] = linspace[0] --> 0.0 
			//i = 1, j = 0 yield col_1[1] = linspace[0] --> 0.0
			col_1[j * 10 + i] = linspace[j];
		}

	}

	//distribute linspace-function in column 2 using a modulo-operator
	for (int i = 0; i < CAPACITY; i++)
	{
		//iteration scheme: col_2[0] = linspace[0 % 10] = linspace[0] --> 0.0
		//col_2[1] = linspace[1 % 10] = linspace[1] --> 0.11
		//...
		//col_2[10] = linspace[10 % 10] = linspace[0] --> 0.0
		col_2[i] = linspace[i % 10];
	}


	//printing the two columns 
	for (int i = 0; i < CAPACITY; i++)
	{
		cout << setprecision(3) << fixed << col_1[i] << "\t" << col_2[i] << endl;
	}

	return 0;
}