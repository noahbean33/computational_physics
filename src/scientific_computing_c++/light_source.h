#include <iostream>
#include <string>

using namespace std;

#ifndef LIGHT_SOURCE_H
#define LIGHT_SOURCE_H

class LightSource {
	
	//public access modifier
public: 

	//Default constructor
	LightSource();

	//Overload constructor
	LightSource(string, int, double);

	//Destructor
	~LightSource();

	//Mutator functions (setters)
	void setType (string);
	void setLuminousEfficacy (int);
	void setPrice (double);

	//Accessor functions (getters)
	string getType() const {
		return strType;
	}

	int getLuminousEfficacy() const {
		return intLuminousEfficacy;
	}

	double getPrice() const {
		return dblPrice;
	}

	//extra-function for calculating the ECE
	double calculateECE();


	//private member variables (private access modifier) 
private:
	string strType; int intLuminousEfficacy; double dblPrice;

};

#endif 