#include "Matrix.h"
#include <iostream>
#include <fstream>
#include <sstream>

using namespace std;

//Overload-constructor
Matrix::Matrix(string p_content)
{
	dataIsSet = false;
	addcontent(p_content);
}

//Accessor-funct.
string Matrix::getName() const 
{
	return m_name;
}

float Matrix::getValue(int row, int col)
{
	return dataarray[row][col];
}


void Matrix::addcontent(string p_contentfile)
{
	m_contentfile = p_contentfile;
	ifstream myFile;
	cout << "Try to open " << m_contentfile << endl << endl;
	myFile.open(m_contentfile);

	if(!myFile)
	{
		cerr << "File cannot be read in\n";
	}
	else 
	{
		char pcBuff[256];
		//read in the whole lines into buffer
		myFile.getline (pcBuff, 256);
		//assign m_name to the string of defined buffersize
		m_name = string(pcBuff);
		string lastColumnLine;
		while (!myFile.eof() )
		{
			myFile.getline (pcBuff, 256);
			char* compareToChar = "    ";
			string line = string(pcBuff);
			//substr.-funct. for extracting the first 4 characters
			string beginLine4Chars = line.substr(0,4);
			if ((beginLine4Chars.compare(compareToChar) == 0) && checkIfStringContainsNumber(line))
			{
				lastColumnLine = line;
			}
		}
		lastColumnLine.erase(lastColumnLine.find_last_not_of(" ") + 1);
		istringstream(lastColumnLine.substr(lastColumnLine.find_last_of(" ") + 1)) >> col;
		cout << "Number of columns: " << col << endl;

		//clear what we stored into our buffer
		myFile.clear();
		myFile.seekg(0, ios::beg);
		myFile.getline (pcBuff, 256);

		row = 0;
		while (!myFile.eof() )
		{
			myFile.getline (pcBuff, 256);
			char* compareToChar = "    ";
			string line = string(pcBuff);
			//substr.-funct. for extracting the first 4 characters
			string beginLine4Chars = line.substr(0,4);
			if (checkIfStringContainsNumber(line))
			{
				//at this stage tmpNumber variable is undefined
				int tmpNumber;
				istringstream (line) >> tmpNumber;
				if (tmpNumber > row) 
				{
					row = tmpNumber;
				}
			}
		}
		cout << "Number of rows: " << row << endl << endl;

		//initialization of dataarray
		dataarray = new float*[row];

		for (int i = 0; i < row; i++)
		{
			dataarray[i] = new float[col];
		}

		myFile.clear();
		myFile.seekg(0, ios::beg);

		int lastColumnNumber, rows;
		while(!myFile.eof() )
		{
			myFile.getline (pcBuff, 256);
			char* compareToChar = "    ";
			string line = string(pcBuff);
			string beginLine4Chars = line.substr(0,4);
			if ((beginLine4Chars.compare(compareToChar) == 0) && checkIfStringContainsNumber(line))
			{
				line.erase(0, line.find_first_not_of(" "));
				line = line.substr(0, line.find_first_of(" "));
				istringstream (line) >> lastColumnNumber;
				
			}
			else 
			{
				if (checkIfStringContainsNumber(beginLine4Chars))
				{
					//deletes the first four characters
					line.erase(0,4);
					istringstream(beginLine4Chars) >> rows;
					float temp = -1;
					//for-loop for counting the amount of float values on a current line
					for (int colinc = 0; colinc < getNumberCount(line); colinc++)
					{

						istringstream (line.substr(colinc * 13,13)) >> temp;
						//in the first run: colinc = 0	
						//dataarray[1-1][1-1 + 0] ...[1-1][1-1 + 4]
						//in the second run: colinc = 0 
						//dataarray[2-1][1-1 + 0] ...[2-1][1-1 + 4]
						//iterating up to row number 10:
						//dataarray[10-1][1-1 + 0] ...[10-1][1-1 + 4]
						//dataarray[9][0] ... [9][4]
						dataarray[rows-1][lastColumnNumber-1 + colinc] = temp;
					}
				}
			}
		}
	}
}

Matrix::~Matrix()
{
	if (dataIsSet) 
	{
		for (int i = 0; i < row; i++)
		{
			delete []dataarray[i];
		}
		for (int j = 0; j < col; j++)
		{
			delete []dataarray[j];
		}
		delete []dataarray;
	}

}

bool Matrix::checkIfStringContainsNumber(string p_tempstr)
{
	if ((p_tempstr.find("0") == string::npos) &&
		(p_tempstr.find("1") == string::npos) &&
		(p_tempstr.find("2") == string::npos) &&
		(p_tempstr.find("3") == string::npos) &&
		(p_tempstr.find("4") == string::npos) &&
		(p_tempstr.find("5") == string::npos) &&
		(p_tempstr.find("6") == string::npos) &&
		(p_tempstr.find("7") == string::npos) &&
		(p_tempstr.find("8") == string::npos) &&
		(p_tempstr.find("9") == string::npos))
	{
		return false;
	}
	else
	{
		return true;
	}
}

int Matrix::getNumberCount(string p_linestring)
{
	int numberCount = 0;
	bool isInNumber = false;
	for (unsigned int i = 0; i < p_linestring.length(); i++)
	{
		char tmp = (p_linestring.c_str())[i];
		switch (tmp)
		{
		case '0':
		case '1':
		case '2':
		case '3':
		case '4':
		case '5':
		case '6':
		case '7':
		case '8':
		case '9':
		case '+':
		case '-':
		case '.':
		case 'e':
			if (isInNumber == false)
			{
				numberCount++;
				isInNumber = true;
			}
			break;
		default:
			isInNumber = false;
			break;
		}
	}
	return numberCount;
}
