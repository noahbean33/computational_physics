#include "bmr_calculator.h"

BmrCalculator::BmrCalculator() {
	int intAge = 0;
	int intHeight = 0;
	double dblWeight = 0.0;
}

BmrCalculator::BmrCalculator (int age, int height, double weight) {
	intAge = age;
	intHeight = height;
	dblWeight = weight;

}

BmrCalculator::~BmrCalculator() {

}

void BmrCalculator::setAge (int age) {
	intAge = age;
}

void BmrCalculator::setHeight (int height) {
	intHeight = height;
}

void BmrCalculator::setWeight (double weight) {
	dblWeight = weight; 
}

//Harris-Benedict formula for calculating the BMR of a male
double BmrCalculator::calculateBMRMale () {
	return (66 + (13.8 * dblWeight) + (5.0 * intHeight) - (6.8 * intAge) );
}

//Harris-Benedict formula for calculating the BMR of a female
double BmrCalculator::calculateBMRFemale () {
	return (655 + (9.5 * dblWeight) + (1.9 * intHeight) - (4.7 * intAge) );
}