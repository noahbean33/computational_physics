/* Author    : Okan Koeksal
 Copyright   : Free to modify 
 This code has been written by Okan Koeksal in the course of the C++ tutorials
 If you have any questions or find bugs, feel free to mail me "koeksal.okan@gmail.com"*/

#include <iostream>
#include <string>
#include <iomanip>
#include <fstream>
#include <vector>

using namespace std;


int main () {

	//boolean-variable to decide which section of the csv-file should be read in
	bool foundStringDelimiter = false;
	//vector of vectors for holding the data
	vector<vector<double> > data;
	ifstream myFile("C:\\Temp\\data_csv.txt", ios::in);
	if (myFile.is_open() )
	{
		cout << "Successfully opened the csv-file!" << endl;

		while (!myFile.eof()) 
		{
			//hold the read in line
			string line;
			//read line from csv-file into line
			getline(myFile, line);

			if(foundStringDelimiter)
			{
				if (line == "Values #2")
				{
					//here the relevant part of the csv-file ends
					foundStringDelimiter = false;
				}
				else
				{
					//array for holding the position of the comma
					int commaPos[2];
					int posIndex = 0;

					for (int i = 0; i < line.length(); i++)
					{
						if (line[i] == ',')
						{
							commaPos[posIndex] = i; 
							posIndex++;
						}

					}
				//temporary variables for the strings representing x, y and z points
				string x_string, y_string, z_string;
				double x_temp, y_temp, z_temp;

				//extracting values by making use of the substr-function
				x_string = line.substr(0, commaPos[0]);
				y_string = line.substr(commaPos[0] + 1, commaPos[1] - commaPos[0] - 1);
				z_string = line.substr(commaPos[1] + 1);

				//conversion of string into a double datatype
				x_temp = atof(x_string.c_str());
				y_temp = atof(y_string.c_str());
				z_temp = atof(z_string.c_str());

				//temporary double vector variable
				vector<double> temp;
				//add x_temp, y_temp as well as z_temp to the temporary vector
				temp.push_back(x_temp);
				temp.push_back(y_temp);
				temp.push_back(z_temp);
				//push the temporary vector onto the vector of vectors 
				data.push_back(temp);

				}

			}
			else
			{
				if (line == "Values #1")
				{
					//here the relevant part in our csv-file starts
					foundStringDelimiter = true;
				}
			}

		}

	}
	else 
	{
		cout << "Unable to open the csv-file!" << endl;
	}

	//close the csv-file
	myFile.close();

	//printing the double values
	for (int i = 0; i < data.size(); i++)
	{
		cout << fixed << setprecision(2) << "x = " << data[i][0] << ", y = " << data[i][1] << ", z = " << data[i][2] << endl;
	}




	return 0;
}