/* Author    : Okan Koeksal
 Copyright   : Free to modify 
 This code has been written by Okan Koeksal in the course of the C++ tutorials
 If you have any questions or find bugs, feel free to mail me "koeksal.okan@gmail.com"*/

#include <iostream>
#include <string>

const int CAPACITY = 30; 

using namespace std; 

string studentName[CAPACITY];
string subject[CAPACITY];
int studentScore[CAPACITY];

void studentInfo(string, string, int);
void lookUpStudent(); 
void addStudent();
void editStudent();
void deleteStudent();

int main() {

	int menuChoice = 0;

	do {
		cout << "\tWelcome To The Main Menu For Transcripts Of Students\t" << endl;
		cout << "1. Look up the scores of any student!" << endl;
		cout << "2. Add the name of student, subject and the exam score!" << endl;
		cout << "3. Edit the name of student, subject and the exam score!" << endl;
		cout << "4. Delete the name of student, subject and the exam score!" << endl;
		cout << "5. Exit!" << endl;

		cout << "Please enter your choice: ";
		cin >> menuChoice;

		while(menuChoice < 1 || menuChoice > 5)
		{
				cout << "Your choice has to be in the range of 1-4" << endl;
				cout << "Please re-enter your choice: ";
				cin >> menuChoice;
		}

		switch (menuChoice)
		{
		case 1:
			lookUpStudent();
			break;
		case 2:
			addStudent();
			break;
		case 3:
			editStudent();
			break;
		case 4:
			deleteStudent();
			break;
		default:
			cout << "You exited the program!" << endl;
			break;
		}


	}while(menuChoice != 5);


return 0; 
}

void studentInfo(string studentName, string subject, int studentScore)
{
	cout << "Student's name: " << studentName << endl;
	cout << "Subject: " << subject << endl;
	cout << "Student's exam score: " << studentScore << endl;

}


void lookUpStudent()
{
	bool foundRecord = false;
	int arrIndex = 0;
	string searchStudent;
	cin.ignore();

	cout << "Enter the student's name: ";
	getline(cin, searchStudent);

	while(!foundRecord && arrIndex < CAPACITY)
	{
		if (studentName[arrIndex] == searchStudent)
		{
			foundRecord = true;
		}
		else 
			arrIndex++;
	}

	if (foundRecord)
	{
		studentInfo(studentName[arrIndex], subject[arrIndex], studentScore[arrIndex]);
		cout << endl << endl;
	}
	else 
		cout << "The name of the student could not be found!" << endl;

}

void addStudent() 
{
	bool foundRecord = false;
	int arrIndex = 0;

	while(!foundRecord && arrIndex < CAPACITY)
	{
		if (studentName[arrIndex] == "")
		{
			foundRecord = true;
		}
		else 
			arrIndex++;
	}
	
	if(foundRecord)
	{
		cin.ignore();
		cout << "Please type in the name of the student: ";
		getline(cin, studentName[arrIndex]);

		cout << "Please type in the subject in which the exam was taken: ";
		getline(cin, subject[arrIndex]);

		cout << "Please type in the score: ";
		cin >> studentScore[arrIndex];
		cout << endl << endl;
	}
	else 
		cout << "The size of max. students is exceeded, no more entries allowed" << endl;

}

void editStudent()
{
	bool foundRecord = false;
	int arrIndex = 0;
	int submenuChoice = 0;
	string searchStudent; 

	cin.ignore();
	cout << "Enter the student's name to be edited: ";
	getline(cin, searchStudent);

	while(!foundRecord && arrIndex < CAPACITY)
	{
		if (studentName[arrIndex] == searchStudent)
		{
			foundRecord = true;
		}
		else
			arrIndex++;
	}

	if (foundRecord)
	{
		studentInfo(studentName[arrIndex], subject[arrIndex], studentScore[arrIndex]);

		do {
			cout << "You might want to edit the name of the student by chosing: " << endl;
			cout << "1. Student's name" << endl;
			cout << "2. Subject in which the exam was taken" << endl;
			cout << "3. Score of the exam" << endl;
			cout << "4. Exit" << endl;
			cin >> submenuChoice;

			while(submenuChoice < 1 || submenuChoice > 4)
			{
				cout << "Your choice has to be in the range of 1-4" << endl;
				cout << "Please re-enter your choice: ";
				cin >> submenuChoice;
			}

			switch (submenuChoice)
			{
			case 1:
				cout << "Current name of the student: ";
				cout << studentName[arrIndex] << endl;
				cin.ignore();
				cout << "Enter the new name of the student: ";
				getline(cin, studentName[arrIndex]);
				break;
			case 2:
				cout << "Current subject: ";
				cout << subject[arrIndex] << endl;
				cin.ignore();
				cout << "Enter the new subject: ";
				getline(cin, subject[arrIndex]);
				break;
			case 3:
				cout << "Current score of the exam: ";
				cout << studentScore[arrIndex] << endl;
				cin.ignore();
				cout << "Enter the new score: ";
				cin >> studentScore[arrIndex];
				break;
			case 4:
				cout << "You exited the program!" << endl;
				break;
			}


		}while(submenuChoice != 4);

	}
	else
		cout << "The student you are looking for is not in the list!" << endl;

}

void deleteStudent() 
{
	bool foundRecord = false;
	int arrIndex = 0;
	int menuChoice = 0;
	string searchStudent;
	char confirmDeletion;
	
	cin.ignore();
	cout << "Enter the student's name to be deleted: ";
	getline(cin, searchStudent);

	while(!foundRecord && arrIndex < CAPACITY)
	{
		if (studentName[arrIndex] == searchStudent)
		{
			foundRecord = true;
		}
		else 
			arrIndex++;
	}

	if (foundRecord)
	{
		studentInfo(studentName[arrIndex], subject[arrIndex], studentScore[arrIndex]);
		cout << "Are you sure that you want to delete this student from the list, press y (yes) or n (no)?";
		cin >> confirmDeletion;

		if (confirmDeletion == 'Y' || confirmDeletion == 'y')
		{
			studentName[arrIndex] = "";
			subject[arrIndex] = "";
			studentScore[arrIndex] = 0;
		}

	}
	else 
		cout << "The student you are looking for is not in the list!" << endl << endl;
}
