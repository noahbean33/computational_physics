#include "light_source.h"

LightSource::LightSource() {
	string strType = "";
	int intLuminousEfficacy = 0;
	double dblPrice = 0.0;
}

LightSource::LightSource (string types, int efficacies, double prices) {
	setType(types);
	setLuminousEfficacy(efficacies);
	setPrice(prices);
}

LightSource::~LightSource() {

}


void LightSource::setType (string types) {
	if (types.length() > 0) {
		strType = types;
	}
	else {
		cout << "Data could not be read, might have a different format for type of light" << endl;
		exit(1);
	}
}

void LightSource::setLuminousEfficacy (int efficacies) {
	if (efficacies > 0) {
		intLuminousEfficacy = efficacies;
	}
	else {
		cout << "Data could not be read, might have a different format for luminous efficacy" << endl;
		exit(1);
	}
}

void LightSource::setPrice (double prices) {

	if (prices > 0 ) {
		dblPrice = prices;
	}
	else {
		cout << "Data could not be read, might have a different format for price" << endl;
		exit(1);
	}

}


double LightSource::calculateECE() {
	return ( (intLuminousEfficacy / 350.0) * 100);
}
