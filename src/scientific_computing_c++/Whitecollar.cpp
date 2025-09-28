#include "Whitecollar.h"

int Whitecollar::companyBalance = 100000;

Whitecollar::Whitecollar() {
	hourlywage = 0;
	bonus = 0;
}

Whitecollar::Whitecollar(int newHourlywage, int newBonus, int newHours, int newWorkers) 
	: Bluecollar(newHours, newWorkers) {
		hourlywage = newHourlywage;
		bonus = newBonus; 
}

Whitecollar::~Whitecollar() {

}

int Whitecollar::getHourlywage() const {
	return hourlywage;
}

int Whitecollar::getBonus() const {
	return bonus;
}

void Whitecollar::setHourlywage(int newHourlywage) {
	hourlywage = newHourlywage;
}

void Whitecollar::setBonus(int newBonus) {
	bonus = newBonus;
}

int Whitecollar::getSalaryBonus() const {
	return (getSalary(hourlywage) + bonus);
}

void Whitecollar::printSalary() const {
	cout << "Basic wage for blue collar workers: " << getSalary(hourlywage) << endl;
	cout << "Basic wage including bonus for white collar workers: " << getSalaryBonus() << endl;
	companyBalance -= (getSalaryBonus() + getSalary(hourlywage));
	cout << "Company balance after payment of employees for a week's work: " << companyBalance << endl;
}