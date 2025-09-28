/* Author    : Okan Koeksal
 Copyright   : Free to modify 
 This code has been written by Okan Koeksal in the course of the C++ tutorials
 If you have any questions or find bugs, feel free to mail me "koeksal.okan@gmail.com"*/

#include <iostream>
#include <ctime>

using namespace std; 


int main () 
{
	int randNumber1, randNumber2;
	int userAnswer, compAnswer;
	
	bool wrongAnswer;

	//seed the random number from the time machine
	srand (time(0));

	//generate two random numbers between 1 and 19 
	randNumber1 = 1 + rand() % 20;
	randNumber2 = 1 + rand() % 20;

	compAnswer = randNumber1 * randNumber2;

	cout << "Please type in the result of the product of " << randNumber1 << " and " << randNumber2 << ": ";

	do 
	{			
		cin >> userAnswer;

		if (userAnswer != compAnswer)
		{
			cout << "The answer is wrong, please try again! ";
			wrongAnswer = true;
		}
		else if (userAnswer == compAnswer)
		{
			cout << "The answer is correct!" << endl;
			cout << "The product of " << randNumber1 << " x " << randNumber2 << " is: " << compAnswer << endl;
			wrongAnswer = false;
		}

	} while (wrongAnswer == true);

	return 0;
}
