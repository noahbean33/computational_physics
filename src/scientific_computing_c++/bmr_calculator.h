#include <iostream>
#include <string>

using namespace std;

#ifndef BMR_CALCULATOR_H
#define BMR_CALCULATOR_H

class BmrCalculator {

public:
	//Default constructor
	BmrCalculator();
	//Overload constructor
	BmrCalculator(int, int, double);
	//Destructor
	~BmrCalculator();

	//Mutator function
	void setAge (int);
	void setHeight (int);
	void setWeight (double);

	//Accessor functions
	int getAge () const {
		return intAge;
	}

	int getHeight () const {
		return intHeight;
	}

	double getWeight () const {
		return dblWeight;
	}

	//extra-function for calculating the BMR 
	double calculateBMRMale ();
	double calculateBMRFemale ();


private: 
	//Member variables
	int intAge; int intHeight; double dblWeight;


};

#endif