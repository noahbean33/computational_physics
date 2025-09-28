/* Author    : Okan Koeksal
 Copyright   : Free to modify 
 This code has been written by Okan Koeksal in the course of the C++ tutorials
 If you have any questions or find bugs, feel free to mail me "koeksal.okan@gmail.com"*/

#include <iostream>
#include "Whitecollar.h"

using namespace std;



int main () {
	
	int hourlywage, bonus, workers, hours; 

	cout << "Type in the hourly wage for all employees: ";
	cin >> hourlywage;

	cout << "Type in the bonus for managers for one week: ";
	cin >> bonus;

	cout << "Type in the number of all employees: ";
	cin >> workers;

	cout << "Type in the hours for all employees for one week: ";
	cin >> hours;

	Whitecollar mySalary (hourlywage, bonus, workers, hours);
	cout << endl;
	mySalary.printSalary();

	cout << "---------------------------" << endl;
	cout << "Type in the same information for workers working in another division within the company!" << endl;

	cout << "Type in the hourly wage for all employees: ";
	cin >> hourlywage;

	cout << "Type in the bonus for managers for one week: ";
	cin >> bonus;

	cout << "Type in the number of all employees: ";
	cin >> workers;

	cout << "Type in the hours for all employees for one week: ";
	cin >> hours;

	Whitecollar mySalary2 (hourlywage, bonus, workers, hours);
	cout << endl;
	mySalary2.printSalary();

return 0;
}