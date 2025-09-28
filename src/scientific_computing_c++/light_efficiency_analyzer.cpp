/* Author    : Okan Koeksal
 Copyright   : Free to modify 
 This code has been written by Okan Koeksal in the course of the C++ tutorials
 If you have any questions or find bugs, feel free to mail me "koeksal.okan@gmail.com"*/

#include "light_source.h"
#include <iostream>
#include <fstream>
#include <vector>
#include <iomanip>

using namespace std;

void getLightPerformance();
void printECE(vector<LightSource>, double);
double calculateTotalPrice(vector<LightSource>);

int main () {

	getLightPerformance();



return 0;
}

void getLightPerformance() 
{
	ifstream myFile; 
	myFile.open ("C:\\Temp\\lights_values.txt", ios::in);
	if (myFile.fail() )
	{
		cout << "Csv-file could not be opened" << endl;
		exit(1);
	}
	else 
	{
		//holding the string
		string line;
		vector<LightSource> listOfLights;
		//reading the line from the file into line
		while (getline(myFile, line))
		{
			int commaPos[2];
			int posIndex = 0;
			//for-loop serving for finding the position of the two commas in our csv-file
			for (int i = 0; i < line.length(); i++)
			{
				if (line[i] == ',')
				{
					commaPos[posIndex] = i;
					posIndex++;
				}

			}
			//extracting the "type of lights" 
			string types = line.substr(0, commaPos[0]);
			//extracting the "luminous efficacy" and convert the "string" to "int" using the stoi-funct.
			int efficacy = stoi (line.substr(commaPos[0] + 1, commaPos[1] - commaPos[0] - 1));
			//extracting the "price" and convert the "string" to "double" using the atof-funct.
			double prices = atof(line.substr(commaPos[1] + 1).c_str());

			//add type of light, luminous efficacy and prices to our vector by pushing those in our listOfLights
			listOfLights.push_back(LightSource(types, efficacy, prices));
		}
		//closing the file
		myFile.close();
		double total = calculateTotalPrice(listOfLights);
		printECE(listOfLights, total);
	}

}

//calculating the total price of all sorts of lights
double calculateTotalPrice(vector<LightSource> listOfLights) 
{
	double total = 0.0;
	for (int i = 0; i < listOfLights.size(); i++)
	{
		total += listOfLights[i].getPrice();
	}
	return total;
}

//printing the energy conversion efficiency
void printECE(vector<LightSource> listOfLights, double total) 
{
	for (int i = 0; i < listOfLights.size(); i++)
	{
		cout << "Light type #" << (i + 1) << ": " << listOfLights[i].getType();
		cout << ", Luminous efficacy: " << listOfLights[i].getLuminousEfficacy() << " lm/W";
		cout << ", Price: " << listOfLights[i].getPrice() << " Euro";
		cout << ", ECE: " << fixed << setprecision(2) << listOfLights[i].calculateECE() << endl;
	}
	cout << "The total price for the lights is: " << fixed << setprecision(2) << total << " Euro." << endl;

}