#include <iostream>
#include <string>

using namespace std;

#ifndef GUEST_H
#define GUEST_H

class Guest {
public:
	Guest();
	Guest(string, int);
	~Guest();

	//Mutator functions (setters)
	void setName (string);
	void setDays (int);

	//Accessor functions (getters)
	string getName() const;
	int getDays() const;

	//function for calculating the charges
	double calculateCharges();

private: string strName;
		 int intDays;
};




#endif