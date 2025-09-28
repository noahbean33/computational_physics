#include <string>
#include <iomanip>


#ifndef Matrix_H
#define Matrix_H

using namespace std;

class Matrix {
public:
	Matrix();
	Matrix(string);
	~Matrix();

	//parsing of the string-content and filling of array
	void addcontent(string);
	//function for printing the header of the read in file
	string getName() const;
	//function for accessing and printing each element individually on our 2D-array
	float getValue(int, int);


private:
	//declaration and allocation of dataarray as a pointer to pointers
	float** dataarray;
	int row, col;

	//header of the content and content to be read in from file
	string m_name, m_contentfile;

	//ensure that array is initialized and can be accessed
	bool dataIsSet;

	//funct. for checking whether a number is encountered in a specific line
	bool checkIfStringContainsNumber(string);

	//funct. for counting the amount of float values in each row
	int getNumberCount(string);
};



#endif