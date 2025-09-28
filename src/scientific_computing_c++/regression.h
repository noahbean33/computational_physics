/* Author    : Okan Koeksal
 Copyright   : Free to modify 
 This code has been written by Okan Koeksal in the course of the C++ tutorials posted on Youtube 
 If you have any questions or find bugs, feel free to mail me "koeksal.okan@gmail.com"*/
 
#include <iostream>
#include <fstream>
#include <iomanip>
#include <vector>
//accumulate()-funct., inner_product()-funct.
#include <numeric> 
#include <string>
//assert()-funct.
#include <cassert> 

#ifndef regression_H
#define regression_H


using namespace std;

//vector of datatype double
typedef vector<double> array;

class regression 
{
protected:
	//pair of variates x and y
	array x, y;
	//amount of pair of variates
	int n;
	//coefficients a, b
	double a, b;
	//variables for calculating the sum of the squared deviations
	double sx2, sy2, sxy;
public:
	//Default constructor: protected member variables initialized to their NULL state
	regression() {n = 0; a = b = sx2 = sy2 = sxy = 0.0;};
	//virtual functions
	virtual void input(), output();
	//checks whether a regression line exists or not
	bool coefficients();
	//correlation coefficient according to Pearson's formula
	void correlation();
};

void regression::input()
{
	//opens the text file with the values for the linear regression
	ifstream myFile ("C:\\Temp\\linear.txt", ios::in);
	if (myFile.fail()) 
	{
		cout << "File could not be opened!" << endl;
		exit(1);
	}
	else 
	{
		//x and y values
		double x1, y1;
		while (myFile >> x1 >> y1)
		{
			x.push_back(x1);
			y.push_back(y1);
		}
		//amount of pair of variates
		n = x.size();
		cout << n << " pair of variates were read in successfully!" << endl;
	}
	myFile.close();
}

//boolean function returns false if there is no best fit straight line available
bool regression::coefficients()
{
	double sx, sy, mean_x, mean_y;
	sx = sy = mean_x = mean_y = 0.0;
	//sum of x and y values 
	sx = accumulate(x.begin(),x.end(),sx);
	sy = accumulate(y.begin(),y.end(),sy);
	//mean of x and y values
	mean_x = sx / n;
	mean_y = sy / n;
	//sum of x- and y-squared values by using inner_product()-funct.
	//x.begin()*x.begin() + x.end()*0
	sx2 = inner_product(x.begin(),x.end(),x.begin(),sx2);
	sy2 = inner_product(y.begin(),y.end(),y.begin(),sy2);
	//sum of x*y-values
	sxy = inner_product(x.begin(),x.end(),y.begin(),sxy);
	//denominator for b
	double denominator = n*sx2 - sx*sx;
	if (denominator != 0)
	{
		//coefficient b and a
		b = (n*sxy - sx*sy) / denominator;
		//a = sx2*sy - sx*sxy / denominator;
		a = mean_y - b*mean_x;
		return true;
	}
	else
		return false;
}

void regression::correlation()
{
	double correlationCoefficient = sxy  / sqrt(sx2*sy2);
	cout << "Correlation coefficient: r = " << correlationCoefficient << endl;
}

void regression::output()
{
	cout << "Linear regression formula: y = a + b*x" << endl
		 << "a = " << a << endl << "b = " << b << endl << endl;
	cout << fixed << setw(6) << "x" << setw(10) << "y" << setw(10)
		 << "y_calc" << endl;
	for (int i = 0; i < x.size(); i++)
	{
		cout << setw(6) << setprecision(2) << x[i]
			 << setw(10) << setprecision(2) << y[i] 
			 << setw(12) << setprecision(6) << (a + b*x[i]) << endl;
	}
}

class expoRegression : public regression
{
public: 
	expoRegression():regression() {}
	void input(), output();
};


void expoRegression::input()
{
	//opens the text file with the values for the non-linear regression
	ifstream myFile ("C:\\Temp\\exponential.txt", ios::in);
	if (myFile.fail()) 
	{
		cout << "File could not be opened!" << endl;
		exit(1);
	}
	else 
	{
		//x and y values
		double x1, y1;
		while (myFile >> x1 >> y1)
		{
			assert(y1 > 0);
			//y = a*exp(b*x) --> ln(y) = ln(a) + b*x
			y1 = log(y1);
			x.push_back(x1);
			y.push_back(y1);
		}
		//amount of pair of variates
		n = x.size();
		cout << n << " pair of variates were read in successfully!" << endl;
	}
	myFile.close();
}

void expoRegression::output()
{
	cout << "Exponential function: y = a * exp(b*x)" << endl <<
		"Linear form of the regression equation: ln(y) = ln(a) + b*x" << endl
		 << "ln(a) = " << a << endl << "b = " << b << endl << endl;
	cout << fixed << setw(6) << "x" << setw(10) << "y" << setw(10)
		 << "y_calc" << endl;
	for (int i = 0; i < x.size(); i++)
	{
		cout << setw(6) << setprecision(2) << x[i]
			 << setw(10) << setprecision(2) << exp(y[i]) 
			 << setw(12) << setprecision(6) << exp(a + b*x[i]) << endl;
	}
}


#endif