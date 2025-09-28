/* Author    : Okan Koeksal
 Copyright   : Free to modify 
 This code has been written by Okan Koeksal in the course of the C++ tutorials
 If you have any questions or find bugs, feel free to mail me "koeksal.okan@gmail.com"*/

#include <string>
#include <iostream>


using namespace std;

int main() {

	//initialize to empty strings
	string studentsFirstName = "";
	string studentsLastName = "";
	int score = 0;
	int scores[3]; 

	cout << "Please type in your first name: ";
	cin >> studentsFirstName;

	cout << "Please type in your last name: ";
	cin >> studentsLastName;

	studentsFirstName += " " + studentsLastName;

	bool wrongScore;
	int total = 0;
	do {
		for (int i = 0; i < 3; i++) 
		{
			cout << "Please type in the three exam scores: ";
			cin >> scores[i];
			total += scores[i];
			wrongScore = false;

			if (total > 100) 
			{
				cout << "The sum of all parts of the exam must not exceed 100 points, please re-enter the scores!"<< endl;
				wrongScore = true;
				total = 0;
			}
		}
	}while(wrongScore); //keep looping while the total of scores exceed 100 points

	if (total == 100) 
	{
		cout << "Perfect, you passed the exam with an excellent grade!" << endl;
	}

	//else if (total >= 80 && total <= 95) 
	//{
	//	cout << "You passed the exam with a good grade!" << endl;
	//}

	//else if (total >= 80 || total <= 95) 
	//{
	//	cout << "You passed the exam with a good grade!" << endl;
	//}

	//else if (total >= 80 ^ total <= 95) 
	//{
	//	cout << "You passed the exam with a good grade!" << endl;
	//}

	else if (total <= 80 ^ total >= 95) 
	{
		cout << "You passed the exam with a good grade!" << endl;
	}

	else if (total >= 50 && total <= 60)
	{
		cout << "You almost failed and passed the exam with a bad grade!" << endl;
	}
	else 
		cout << "Sorry, but you failed!"<< endl;

	cout << "Full name: " << studentsFirstName << endl;
	cout << "Total score for the exam: " << total << endl;

	for (int i = 0; i < 3; i++)
	{
		cout << "The scores for each exam #" << (i+1) << ": " << scores[i] << " " << endl;
	}
	cout << endl;

	return 0;
}