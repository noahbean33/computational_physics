/* Author    : Okan Koeksal
 Copyright   : Free to modify 
 This code has been written by Okan Koeksal in the course of the C++ tutorials
 If you have any questions or find bugs, feel free to mail me "koeksal.okan@gmail.com"*/
 
#include <iostream>
#include <string>


using namespace std;

enum Figure {rectangle, rightTriangle};

//typedef enum Figure Figure; 

struct FigureProperties 
{
	double sidelengthA, sidelengthB, area;
	string geometry;
};

double areaCalculator (Figure F, double a, double b)
{
	double area;

	switch (F)
	{
	case rectangle:
	//case 0:
		area = a*b;
		break;
	case rightTriangle:
		area = (a*b)/2;
		break;
	}
	return area;
}



int main()
{
	FigureProperties re, rt;
	int choice = 0;
	
	do {
		cout << "1 for rectangle" << endl;
		cout << "2 for right triangle" << endl;
		cout << "3 to exit" << endl;

		cout << "Please choose the figure to calculate the area for: ";
		cin >> choice;

		while (choice < 1 || choice > 3)
		{
			cout << "Your choice has to be in the range of 1-3: ";
			cout << "Invalid choice, please re-enter: ";
			cin >> choice;
		}
		switch (choice)
		{
		case 1:
			cout << "Side lengths a and b? ";
			cin >> re.sidelengthA >> re.sidelengthB;
			re.geometry = "rectangle";
			re.area = areaCalculator(rectangle, re.sidelengthA, re.sidelengthB);
			cout << "Calculated area for the " << re.geometry << " is: " << re.area << endl << endl;
			break;
		case 2:
			cout << "Side lengths a and b? ";
			cin >> rt.sidelengthA >> rt.sidelengthB;
			rt.geometry = "right triangle";
			rt.area = areaCalculator(rightTriangle, rt.sidelengthA, rt.sidelengthB);
			cout << "Calculated area for the " << rt.geometry << " is: " << rt.area << endl;
			break;
		default:
			cout << "You exit the program!" << endl;
		}
	}while(choice != 3);

return 0;
}