/* Author    : Okan Koeksal
 Copyright   : Free to modify 
 This code has been written by Okan Koeksal in the course of the C++ tutorials 
 If you have any questions or find bugs, feel free to mail me "koeksal.okan@gmail.com"*/
 
#include <iostream>
#include <string>

using namespace std;


class carDealer {
protected: 
	int numberOfCars;

public:
	//user-defined default constructor
	carDealer(int number = 0):numberOfCars(number){}
	//virtual Destructor
	virtual ~carDealer(){};
	//virtual function
	virtual int value() const = 0;

};

class Ferrari : public carDealer {
public:
	//price per ordered cars
	static int price;
	Ferrari(int number) : carDealer(number) {cout << "We bought so many Ferraris: " << number << endl;}
	virtual int value() const {return numberOfCars * price;}
};
//set the static int price to 200.000
int Ferrari :: price = 200000;

class Audi : public carDealer {
public:
	static int price;
	Audi(int number) : carDealer(number) {cout << "We bought so many Audis: " << number << endl;}
	virtual int value() const {return numberOfCars * price;}
};
int Audi :: price = 60000;



int main () {

	carDealer* C;
	Ferrari F(2);
	Audi A(4);
	int sum = 0;

	//C is set equal to the address of Ferrari 
	C = &F;
	cout << "The type of the polymorphic object is: " << typeid(*C).name() << endl;
	sum += (*C).value();
	//sum += C -> value();
	C = &A;
	cout << "The type of the polymorphic object is: " << typeid(*C).name() << endl;
	sum += (*C).value();
	cout << "The total price of all ordered cars is: " << sum << " Euros\n";


	return 0;
}