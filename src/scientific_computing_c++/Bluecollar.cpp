#include "Bluecollar.h"

Bluecollar::Bluecollar() {
	hours = 0;
	workers = 0;
}

Bluecollar::Bluecollar(int newHours, int newWorkers) {
	hours = newHours;
	workers = newWorkers;
}

Bluecollar::~Bluecollar() {

}

int Bluecollar::getHours() const {
	return hours;
}

int Bluecollar::getWorkers() const {
	return workers;
}

void Bluecollar::setHours(int newHours) {
	hours = newHours;
}

void Bluecollar::setWorkers(int newWorkers) {
	workers = newWorkers;
}

//the hourlywage variable will be passed in as a parameter and stems from our "Derived class"
int Bluecollar::getSalary(int hourlywage) const {
	return (hours * workers * hourlywage);
}