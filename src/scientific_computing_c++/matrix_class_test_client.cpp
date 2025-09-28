/* Author    : Okan Koeksal
 Copyright   : Free to modify 
 This code has been written by Okan Koeksal in the course of the C++ tutorials
 If you have any questions or find bugs, feel free to mail me "koeksal.okan@gmail.com"*/
 
#include "Matrix.h"
#include <iostream>
#include <string>

using namespace std;

int main() {
    Matrix MatrixValues("C:\\Temp\\MatrixValues.txt");
	//prints the header of the read in file
	cout << MatrixValues.getName() << endl;

	//invoke getValue()-funct. for accessing the elements individually
	cout << "Matrix a_ij: " << endl;
	cout << "a(0,0): " << MatrixValues.getValue(0,0) << endl;
	cout << "a(0,1): " << MatrixValues.getValue(0,1) << endl;
	cout << "a(10,5): " << MatrixValues.getValue(9,4) << endl;
    return 0;
}