#include <iostream>

using namespace std;

//Base class
#ifndef BLUECOLLAR_H
#define BLUECOLLAR_H

class Bluecollar {
public:
	Bluecollar();
	Bluecollar(int, int);
	~Bluecollar();

	int getHours() const;
	int getWorkers() const;

	void setHours(int);
	void setWorkers(int);

	//exta-function accpeting an integer as a parameter
	int getSalary(int) const;

private:
	int hours;
	int workers;
};

#endif