/* Author    : Okan Koeksal
 Copyright   : Free to modify 
 This code has been written by Okan Koeksal in the course of the C++ tutorials
 If you have any questions or find bugs, feel free to mail me "koeksal.okan@gmail.com"*/

#include <iostream>
#include <iomanip>

using namespace std;

//constant pi-value
const double pi = 3.141592; 


float radian (float degree) {
	float r = 0.0f;
	r = degree * pi / 180;
	return r;
}


int main () {

	float r = 0.0f;
	cout << "Conversion of 1 - 360 degrees to radians: " << endl;

	for (int i = 1; i <= 360; i++) 
	{
		r = radian(i);
		cout << i << " Degree equals to " << setprecision(6) << fixed << r << " radians\n";

	}
	return 0;
}
