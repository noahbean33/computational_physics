/* Author    : Okan Koeksal
 Copyright   : Free to modify 
 This code has been written by Okan Koeksal in the course of the C++ tutorials 
 If you have any questions or find bugs, feel free to mail me "koeksal.okan@gmail.com"*/
 
#include <iostream>
//make_heap()/sort_heap()/pop_heap() as well as *min/*max_element() functions
#include <algorithm>
#include <vector>
//needed for ostream_iterator
#include <iterator>

using namespace std;

const int NUM_ELEMENTS = 7;

//overload the ostream-operator for our vector class
ostream& operator << (ostream& stream, const vector<int>& numbers)
{
	//copy()-function: copy(first, last, output_iterator result)
	copy(numbers.begin(), numbers.end(), ostream_iterator<int>(cout, " "));
	return stream;
}



int main() 
{ 
	//vector declaration
	vector<int> numbers;

	cout << "Building the heap: " << endl;
	for (int i = 0; i < NUM_ELEMENTS; i++)
	{
		//adding an element at the end to the vector
		numbers.push_back((i * 17 + 5) % 100);
		//creation of max heap out of a range of elements by transforming vector into a max heap
		make_heap(numbers.begin(), numbers.end());
		cout << numbers << endl;
	}
	cout << endl;

	//sorting in an increasing order via sort_heap()
	cout << "Sorting in ascending order: " << endl;
	//turning a max heap into a range of elements in increasing order
	sort_heap(numbers.begin(), numbers.end());
	cout << numbers << endl << endl;

	int min, max = 0;
	//*min/*max_element(iterator_start, iterator_end)
	min = *min_element(numbers.begin(), numbers.end());
	max = *max_element(numbers.begin(), numbers.end());

	//output the minimum and maximum elements
	cout << "Smallest element is: " << min << endl;
	cout << "Largest element is: " << max << endl << endl;

	//restore heap
	make_heap(numbers.begin(), numbers.end());
	cout << "Reduction of the heap by 1: " << endl;
	for (int j = NUM_ELEMENTS; j > 0; j--)
	{
		//deletion of the top element by invoking the pop_heap() functions
		pop_heap(numbers.begin(), numbers.begin() + j);
		copy(numbers.begin(), numbers.begin() + j, ostream_iterator<int>(cout, " "));
		cout << "--> " << *(numbers.begin() + j - 1);
		cout << endl;
	}
	cout << endl;


	return 0;
}