#include "Guest.h"

const double CHARGES_PER_DAY = 65.0;
const double DISCOUNT_PER_DAY = 5.0;

Guest::Guest() {
	strName = "";
	intDays = 0;
}

Guest::Guest(string name, int days) {
	strName = name;
	intDays = days;
}

Guest::~Guest() {

}

string Guest::getName() const {
	return strName;
}

int Guest::getDays() const {
	return intDays;
}

void Guest::setName (string name) {
	strName = name;
}

void Guest::setDays (int days) {
	intDays = days;
}

double Guest::calculateCharges() {
	double totalInvoice = 0.0;

	if (intDays > 5) {
		totalInvoice = intDays * CHARGES_PER_DAY;
		totalInvoice -= intDays * DISCOUNT_PER_DAY;
	}
	else {
		totalInvoice = intDays * CHARGES_PER_DAY;
	}

	return totalInvoice;
}
