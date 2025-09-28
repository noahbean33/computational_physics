/* Author    : Okan Koeksal
 Copyright   : Free to modify 
 This code has been written by Okan Koeksal in the course of the C++ tutorials
 If you have any questions or find bugs, feel free to mail me "koeksal.okan@gmail.com"*/
 
#include "bmr_calculator.h"
#include <iostream>

using namespace std;



int main () {

	int age;
	int height;
	double weight;

	char input;
	cout << "Please type in the gender of the person you want to calculate the BMR for by typing 'm' or 'f': ";
	cin >> input;

	if (input == 'f')
	{
		cout << "Please type in the age: ";
		cin >> age;

		cout << "Please type in the height [cm]: ";
		cin >> height;

		cout << "Please type in the weight [kg]: ";
		cin >> weight;

		//passing the parameters in using the Overload constructor
		BmrCalculator Female (age, height, weight);
		cout << "The age of the patient is: " << Female.getAge() << endl;
		cout << "The height of the patient is: " << Female.getHeight() << endl;
		cout << "The weight of the patient is: " << Female.getWeight() << endl;
		cout << "The BMR rate of the patient is: " << Female.calculateBMRFemale() << " [kcal]\n";
	}

	else 
	{
		cout << "Please type in the age: ";
		cin >> age;

		cout << "Please type in the height [cm]: ";
		cin >> height;

		cout << "Please type in the weight [kg]: ";
		cin >> weight;

		BmrCalculator Male;
		//passing in the parameters using the Mutator function
		Male.setAge (age);
		Male.setHeight (height);
		Male.setWeight (weight);

		cout << "The age of the patient is: " << Male.getAge() << endl;
		cout << "The height of the patient is: " << Male.getHeight() << endl;
		cout << "The weight of the patient is: " << Male.getWeight() << endl;
		cout << "The BMR rate of the patient is: " << Male.calculateBMRMale() << " [kcal]\n";
	}


return 0;
}