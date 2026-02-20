#include <iostream>
#include <iomanip>
#include <random>
/*Alternatively, one could use GSL
#include <gsl/gsl_rng.h> */

//initialize 2D array of spins randomly with +/-1
void initializeArray (int**, int, int);

//calculate the energy obtained from the interactions between the adjacent pair of spins 
int calculateEnergy (int**, int, int, int);

//function samples the array of spins and computes the average magnetization and average energy per spin
void sampleArray (int**, int, double&, double&);

using namespace std;

int main () {

	int** array;
	int sideLength = 20;
	int numberOfCycles = 10000;
	int frequencySampling = 1000;
	double temperature = 5.0;
	temperature = 1.0 / temperature;

	mt19937_64 gen;
	uniform_int_distribution<int> dist1(0,2);
	uniform_int_distribution<int> dist2(0,19);
	uniform_real_distribution<double> dist3(0,1);

	int numberOfSamples = 0;
	int delta_E;
	double BoltzmannFactor;

	double spin = 0.0, spinSum = 0.0;
	double energy = 0.0, energySum = 0.0;

	//memory allocation for the dynamically created array
	array = new int* [sideLength];
	for (int i = 0; i < sideLength; i++)
	{
		array[i] = new int[sideLength];
	}

	//assign uniform random number generator to a variable
	int randomNumber = dist1(gen);
	initializeArray(array, sideLength, randomNumber);

	for (int counter = 0; counter < numberOfCycles; counter++)
	{
		for (int average = 0; average < sideLength; average++)
		{
			int i = dist2(gen);
			int j = dist2(gen);
			delta_E = calculateEnergy(array, sideLength, i, j);
			BoltzmannFactor = exp(delta_E * temperature);
			double x = dist3(gen);

			if (x < BoltzmannFactor)
			{
				array[i][j] *= -1;
			}
		}
		if (counter % frequencySampling == 0)
		{
			sampleArray(array, sideLength, spin, energy);
			cout << "Loops: " << counter << fixed << setprecision(3) << "\t spin: " << spin << "\t energy: " << energy << endl;
			spinSum += spin;
			energySum += energy;
			numberOfSamples++;
		}
	}

	cout << "The average magnetization is: " << fixed << setprecision(3) << spinSum / numberOfSamples << endl;
	cout << "The average energy per spin is: " << fixed << setprecision(3) << energySum / numberOfSamples << endl;

	//memory deallocation: deleting the dynamically created array
	for (int i = 0; i < sideLength; i++)
	{
		delete []array[i];
		for (int j = 0; j < sideLength; j++)
		{
			delete []array[j];
		}
		delete []array;
	}

	return 0;
}

void initializeArray (int** array, int sideLength, int random)
{
	for (int i = 0; i < sideLength; i++)
	{
		for (int j = 0; j < sideLength; j++)
		{
			array[i][j] = 2 * (random) - 1;
		}
	}
}

int calculateEnergy (int** array, int sideLength, int i, int j)
{
	return (-2 * (array[i][j]) * (array[(i + 1 + sideLength) % sideLength][j]
								 +array[(i - 1 + sideLength) % sideLength][j]
								 +array[(j + 1 + sideLength) % sideLength][i]
								 +array[(j - 1 + sideLength) % sideLength][i]));
}


void sampleArray (int** array, int sideLength, double& spin, double& energy)
{
	spin, energy = 0.0;

	for (int i = 0; i < sideLength; i++)
	{
		for (int j = 0; j < sideLength; j++)
		{
			spin += static_cast<double> (array[i][j]);
			energy -= static_cast<double> (array[i][j] * (array[i][(j + 1) % sideLength]
														+ array[(i + 1) % sideLength][j]));
		}
	}
	spin /= (sideLength * sideLength);
	energy /= (sideLength * sideLength);
}
