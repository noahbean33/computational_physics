#include "Bluecollar.h"
#include <iostream>

//Derived class
#ifndef WHITECOLLAR_H
#define WHITECOLLAR_H

class Whitecollar : public Bluecollar {
public:
	Whitecollar();
	Whitecollar(int, int, int, int);
	~Whitecollar();

	int getHourlywage() const;
	int getBonus() const;

	void setHourlywage(int);
	void setBonus(int);

	int getSalaryBonus() const;
	void printSalary() const;

private:
	int hourlywage, bonus;
	static int companyBalance;
};

#endif