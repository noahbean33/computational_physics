/* Author    : Okan Koeksal
 Copyright   : Free to modify 
 This code has been written by Okan Koeksal in the course of the C++ tutorials
 If you have any questions or find bugs, feel free to mail me "koeksal.okan@gmail.com"*/

#include "Guest.h"
#include <iostream>
#include <string>
#include <vector>

using namespace std;

void fillVectorsOfObjects (vector<Guest>&);
void printVectorsOfObjects (vector<Guest>&);

int main () {
	
	vector<Guest> daysOfReservations;
	fillVectorsOfObjects (daysOfReservations);
	printVectorsOfObjects (daysOfReservations);

return 0;
}

void fillVectorsOfObjects (vector<Guest>& newDaysOfReservations) {
	string name;
	int days;
	int numOfGuests;

	cout << "Please type in how many guests are going to reserve a room: ";
	cin >> numOfGuests;

	for (int i = 0; i < numOfGuests; i++) {
		cout << "Guest name: ";
		cin >> name;
		cout << "Reserved days: ";
		cin >> days;

		//push the newly created object onto the vector "newDaysOfReservations" by utilizing the Overload constructor
		newDaysOfReservations.push_back(Guest(name, days));
	}

}

void printVectorsOfObjects (vector<Guest>& newDaysOfReservations) {

	for (int i = 0; i < newDaysOfReservations.size(); i++) 
	{
		cout << "-------------------------------------" << endl;
		cout << "Guest name: " << newDaysOfReservations[i].getName() << endl;
		cout << "Reserved days: " << newDaysOfReservations[i].getDays() << endl;
		cout << "Charges the be paid: " << newDaysOfReservations[i].calculateCharges() << endl;
		cout << endl;
	}

}
