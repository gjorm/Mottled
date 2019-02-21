#include <iostream>
#include <vector>
#include <algorithm>
#include <cstdlib>
#include <stdio.h>
#include <fstream>
#include <sstream>
#include <cctype>
#include <cmath>
#include <cstdlib>
#include <omp.h>
#include <ctime>

// defines
#define MIN_PATTERN_WIDTH 1
#define MIN_PATTERN_HEIGHT 3
#define MAX_PATTERN_WIDTH 1
#define MAX_PATTERN_HEIGHT 4


#define e 2.71828

#define TESTED_DRAW 100 // 0 will attempt to predict the next future draw. A positive number will test N predictions at the end of the dataset
#define EN_LOC_AMP false
#define EN_TIMEPLUS false
#define EN_STANDARDIZE_WEIGHTS false //weight standardisation enabled
#define EN_NORMALIZE_DATA true //normalize the data, per column values, to the value range [0,1]. If using relative dataset, do not enable

const double ACCURACY = 0.76; // this value indicates how similar a pattern must be to another, to consider it to be a match

using namespace std;

// struct used to store the patterns
typedef struct _pattern {
public:
	int width, height;
	double image[MAX_PATTERN_HEIGHT][MAX_PATTERN_WIDTH];
	double score;
	int indices[MAX_PATTERN_WIDTH]; // index linking to the row position where the pattern was found
	double recent;
} pattern;

// score struct for keeping track of which scores belong long to which number
typedef struct _score {
public:
    double weight;
    int num;
	vector<double> vals;
	vector<double> wts;
	bool cald = false;

	void CalcWeight() {
		weight = 0;
		double wtMean = 0;
		for (int i = 0; i < (int)vals.size(); i++) {
			weight += (vals[i] * wts[i]);
			wtMean += wts[i];
		}

		weight = weight / wtMean;
		cald = true;
	}

	void Clear() {
		vals.clear();
		wts.clear();
	}
} score;

//prototypes
int ListContains(const pattern &bar, vector<pattern> &p);
bool UpperMatches(const pattern &a, const pattern &b);
bool PatternMatches(const pattern &a, const pattern &b);
bool CompVals(double a, double b);
double GetAcc(double a, double b);
long long Sigmoid(long long x, long long mid, long long highest);
pattern PullFromDataset(int line, int offset, int width, int height, const vector<vector<double> > &data);
bool IndicesMatch(const pattern &foo, const pattern &bar);


//overloaded operators
bool operator == (const pattern& left, const pattern& right) {
	return PatternMatches(left, right);
}

bool operator < (const pattern& left, const pattern& right) {
	return left.score < right.score;
}

bool operator < (score left, score right) {
	if (left.cald == false)
		left.CalcWeight();
	if (right.cald == false)
		right.CalcWeight();

	return left.weight < right.weight;
}

int main() {

	std::cout.precision(6);
	std::cout.setf(std::ios::fixed);

	string out = "";
	time_t before, after;

	//data arrays
	vector<vector<double> > dataset;
	vector<double> row;
	vector<double> min, max;

	// getting time
	before = time(NULL);

	/*
	
	FILE INPUT AND PARSING
	
	*/

	cout << "Reading File Data..." << endl;
	string fileName = "allForexData.csv";
	bool fileRead = true;
	ifstream fileInput(fileName);
	if (fileInput.bad()) {
		cout << "Could not open file: " << fileName << endl;
		fileRead = false;
	}

	//load data into a vector of strings
	string lineS = "";
	vector<string> preData;
	if (fileRead) {
		while (getline(fileInput, lineS)) {

			if (!lineS.empty()) {
				lineS += ','; //add a comma at the end of the string to facilitate parsing
				preData.push_back(lineS);
			}
		}
	}

	cout << "Number of Lines in File: " << preData.size() << endl;

	//parse data

	int numCols = 0;
	cout << "Parsing File Data..." << endl;
	string preWord = "";
	string sampDate = "";
	int rst = 0, day;
	tm timeIn;
	double val;
	for (int i = 0; i < (int)preData.size(); i++) {
		rst = 0;
		day = -1;
		preWord = "";
		//pre load foo with line
		for (int j = 0; j < (int)preData[i].size(); j++) {
			if (preData[i][j] != ',') {
				preWord += preData[i][j];
			}
			else {
				//parse date into a day of week for cleaning up (this is because the data include weekends in which the price stays the same)
				if (rst == 0) {
					rst++;

					timeIn.tm_sec = 0;
					timeIn.tm_min = 0;
					timeIn.tm_hour = 0;

					sampDate = preWord[0];
					sampDate += preWord[1];
					timeIn.tm_mday = stoi(sampDate);
					sampDate = preWord[3];
					sampDate += preWord[4];
					timeIn.tm_mon = stoi(sampDate) - 1;
					sampDate = preWord[6];
					sampDate += preWord[7];
					sampDate += preWord[8];
					sampDate += preWord[9];
					timeIn.tm_year = stoi(sampDate) - 1900;
					time_t timeOut = mktime(&timeIn);

					day = timeIn.tm_wday;

					preWord = "";
				}
				else {
					val = stod(preWord);
					row.push_back(val);

					//record min and max values for normalization on every column
					if (EN_NORMALIZE_DATA) {
						if (i > 0) {
							if (min[rst - 1] > val)
								min[rst - 1] = val;
							if (max[rst - 1] < val)
								max[rst - 1] = val;
						}
						else {
							min.push_back(val);
							max.push_back(val);
						}
					}

					preWord = "";
					rst++;
				}
				
			}
		}

		//check for the day of the week and dont add to dataset if its not friday or saturday
		if (true) { //day != 5 && day != 6

			if (i > 0 && (int)row.size() != numCols) {
				cout << "Dimension error at line #: " << i << endl;
			}
			numCols = (int)row.size();

			
			dataset.push_back(row);
		}

		row.clear();
	}

	int NUM_LINES = (int)dataset.size();
	int NUM_NUMBERS = numCols;

	vector<vector<double>> datasetRel(NUM_LINES, vector<double>(NUM_NUMBERS, 0));

	cout << "Number of Lines in Parsed Dataset: " << dataset.size() << endl;
	
	/*
	
	Normalize Data
	
	*/
	if (EN_NORMALIZE_DATA) {
		cout << "MIN: ";
		for (int i = 0; i < (int)min.size(); i++) {
			cout << min[i] << " ";
		}
		cout << endl << "MAX: ";
		for (int i = 0; i < (int)max.size(); i++) {
			cout << max[i] << " ";
		}
		cout << endl;

		for (int i = 0; i < (int)dataset.size(); i++) {
			//make sure width of the dataset and the min/max vectors are the same
			if (dataset[i].size() == min.size() && dataset[i].size() == max.size()) {
				for (int j = 0; j < (int)dataset[i].size(); j++) {
					dataset[i][j] = abs(dataset[i][j] - min[j]) / (max[j] - min[j]);
				}
			}
			else {
				cout << "Error: Couldn't normalize data as dataset width doesnt match min or max size: " << dataset[i].size() << " " << min.size() << " " << max.size() << " at: " << i << endl;
				break;
			}
		}
	}

	// Initialise relative dataset where the data is relative to the previous rows value
	for (int i = 1; i < (int)dataset.size(); i++) {
		for (int j = 0; j < (int)dataset[i].size(); j++) {
			datasetRel[i][j] = dataset[i][j] - dataset[i - 1][j];
		}
	}


	/*
	
	PATTERN FINDING
	
	*/
	cout << "Pattern Finding..." << endl;
	vector<pattern> patternList;

	int numThreads;

#pragma omp parallel shared(patternList)
	{
		if (omp_get_thread_num() == 0)
			numThreads = omp_get_num_threads();

		pattern foo;
		vector<pattern> _patternList;
		int bip = 0;
		// first two for loops determine pattern limits, the smallest of which is 1x2
		for (int width = MIN_PATTERN_WIDTH; width <= MAX_PATTERN_WIDTH; width++) {
			if (omp_get_thread_num() == 0)
				cout << width << " of " << MAX_PATTERN_WIDTH << "... ";

			for (int height = MIN_PATTERN_HEIGHT; height <= MAX_PATTERN_HEIGHT; height++) {

				foo.width = width;
				foo.height = height;

#pragma omp for
				//next go through, line by line, column by column with the pattern
				for (int line = 1; line <= ((NUM_LINES - height) - TESTED_DRAW); line++) {//line += MAX_PATTERN_SIZE

					for (int col = 0; col <= NUM_NUMBERS - width; col++) { //col += MAX_PATTERN_SIZE

						foo.score = 1;
						foo.recent = (double)line;

						//pull in the pattern from the dataset
						for (int i = 0; i < width; i++) {
							//keep track of original row position...
							foo.indices[i] = col + i;
							for (int j = 0; j < height; j++) {
								//load up foo
								foo.image[j][i] = datasetRel[line + j][col + i];
							}
						}

						// if the pattern is not a match or the vector is empty, add it to the vector
						bip = 0;
						bip = ListContains(foo, _patternList);

						if (bip == -1) {

							_patternList.push_back(foo);

						}
						else { // the list has a match and returns the index of the pattern in the vector, so increment its score
							_patternList[bip].score += 1;
						}
					}
				}
			}
		}

#pragma omp critical 
		{
			patternList.insert(patternList.end(), _patternList.begin(), _patternList.end());
		}
	}

	cout << endl << "Number of Threads: " << numThreads << endl;

	// sort the patternlist before printing
	sort(patternList.begin(), patternList.end());

	/*

	Standardize the Scores

	*/

	if (EN_STANDARDIZE_WEIGHTS) {
		double mean = 0, sqr = 0, msqrd = 0;

		for (int i = 0; i < (int)patternList.size(); i++) {
			mean += (double)patternList[i].score;
		}
		mean = mean / patternList.size();

		for (int i = 0; i < (int)patternList.size(); i++) {
			sqr = patternList[i].score - mean;
			msqrd += pow(sqr, 2.0);
		}
		msqrd = msqrd / patternList.size();

		double standDist = sqrt(msqrd);

		for (int i = 0; i < (int)patternList.size(); i++) {
			patternList[i].score = (double)(patternList[i].score - mean) / standDist;
		}
	}
	

	cout << endl;

	cout << "Number of Patterns Found: " << patternList.size() << endl << endl;
	

	pattern foo;
	ofstream output;
	//write the pattern data to file for checking
	out = "";
	output.open("MottledPatternsFound.txt");
	for (int i = 0; i < (int)patternList.size(); i++) {
		foo = patternList[i];

		for (int j = 0; j < foo.height; j++) {
			for (int k = 0; k < foo.width; k++) {
				out += to_string(foo.image[j][k]) + " ";
			}
			out += "\n";
		}
		out += to_string(foo.score) + "\n\n";
	}

	output << out;
	output.close();
	
	//print the last few values in the dataset

	double disp;
	for (int h = NUM_LINES - 10; h < NUM_LINES; ++h) {
		for (int i = 0; i < NUM_NUMBERS; i++) {
			//Denormalize first
			if (EN_NORMALIZE_DATA) {
				disp = (dataset[h][i] * (max[i] - min[i])) + min[i];
			}
			else {
				disp = dataset[h][i];
			}
			//then print
			cout << disp << " ";
		}

		cout << endl;
	}
	cout << endl;

	/*
	
	PREDICTION
	
	*/
	vector<score> prediction;
	string testS = "";
	score blegh;
	blegh.Clear();
	for (int i = 0; i < NUM_NUMBERS; i++) {	
        blegh.num = i;
        blegh.weight = 0;
        prediction.push_back(blegh);
	}

	pattern bar;
	int line;
	int posMatches, avPosMat = 0, minPosMat = 100000, maxPosMat = 0, failUp = 0, failDown = 0;
	double dispBack = 0, pCtr = 0, tpCtr = 0;
	double acc = 0, totalAcc = 0, extra = 1, timePlus = 1;
	vector<double> grAcc;

	for (int tests = (NUM_LINES - TESTED_DRAW); tests <= NUM_LINES; tests++) {
		
		for (int clr = 0; clr < (int)prediction.size(); clr++) {
			prediction[clr].Clear(); //initially clear
			// to prevent nan output predictions, always start with a baseline of the previous rows information
			prediction[clr].vals.push_back(dataset[tests - 1][clr]); // and re initialize with the previous rows value
			prediction[clr].wts.push_back(1); // and reinitialize with a weight of 1
		}
		posMatches = 0;
		for (int i = 0; i < (int)patternList.size(); i++) {

			foo = patternList[i];

			bar.width = foo.width;
			bar.height = foo.height;

			//next go through column by column with the pattern, testing for a match at the end of the dataset
			for (int col = 0; col <= NUM_NUMBERS - foo.width; col++) {
				line = (tests - foo.height) + 1;

				//pull in the pattern from the dataset
				for (int i = 0; i < foo.width; i++) {
					//pull in the indices as well
					bar.indices[i] = col + i;
					for (int j = 0; j < (foo.height - 1); j++) {
						bar.image[j][i] = datasetRel[line + j][col + i];
					}
				}

				// if the upper portion of the pattern from the pattern list matches whats on the dataset, put the last row of the pattern into the prediction val and the respective weight of the pattern into the wts
				if (UpperMatches(foo, bar)) {
					posMatches++;

					//testing out if original pattern has a match in an original column spot, give it a little more score
					if (EN_LOC_AMP) {
						if (IndicesMatch(bar, foo)) {
							extra = 2;
						}
						else {
							extra = 1;
						}
					}
					else {
						extra = 1;
					}

					//giving slightly more weight to newer patterns
					if (EN_TIMEPLUS) {
						timePlus = 1 + (foo.recent / (double)NUM_LINES);
					}
					else {
						timePlus = 1;
					}

					for (int k = 0; k < foo.width; k++) {
						prediction[col + k].vals.push_back(dataset[tests - 1][col + k] + foo.image[foo.height - 1][k]);
						prediction[col + k].wts.push_back(foo.score * extra * timePlus);
					}
				}


			}
		}

		/*

		Calculate the Prediction Weights and DeNormalize the Prediction if needed

		*/
		for (int i = 0; i < (int)prediction.size(); i++) {
			//make sure width of the prediction and the min/max vectors are the same (this should always check out)
			prediction[i].CalcWeight();

			//Denormalize
			if (EN_NORMALIZE_DATA) {
				prediction[i].weight = (prediction[i].weight * (max[i] - min[i])) + min[i];
			}
		}


		/*

		DISPLAY RESULTS

		*/

		cout << endl << "Positive Matches for Prediction Found: " << posMatches << endl;
		avPosMat += posMatches;
		if (posMatches < minPosMat)
			minPosMat = posMatches;

		if (posMatches > maxPosMat)
			maxPosMat = posMatches;

		string pOrF;

		if (tests < NUM_LINES) {
			testS = "Test #" + to_string(NUM_LINES - tests) + " ";
		}
		else {
			testS = ">>>>>>> Future ";
		}

		cout << testS << "Prediction: " << endl;
		for (int i = 0; i < (int)prediction.size(); i++) {
			cout << prediction[i].weight << " ";
		}
		cout << endl;

		//calculate accuracy
		if (tests < NUM_LINES && TESTED_DRAW > 0) {
			pOrF = "";
			pCtr = 0;
			cout << "Actual:" << endl;
			for (int i = 0; i < NUM_NUMBERS; i++) {
				if (EN_NORMALIZE_DATA) {
					disp = (dataset[tests][i] * (max[i] - min[i])) + min[i];
					dispBack = (dataset[tests - 1][i] * (max[i] - min[i])) + min[i];
				}
				else {
					disp = dataset[tests][i];
					dispBack = dataset[tests - 1][i];
				}
				cout << disp << " ";

				//determine basic up vs down pass or fail...
				if (prediction[i].weight <= dispBack) {
					if (disp <= dispBack) {
						pOrF += "D-Pass-- ";
						pCtr += 1;
					}
					else {
						pOrF += "D-Fail-- ";
						failDown++;
					}	
				}
				else {
					if (disp > dispBack) {
						pOrF += "U-Pass-- ";
						pCtr += 1;
					}
					else {
						pOrF += "U-Fail-- ";
						failUp++;
					}
				}
			}
			cout << endl;
			cout << pOrF << endl;

			
			acc = 0;
			
			for (int i = 0; i < (int)prediction.size(); i++) {
				if (EN_NORMALIZE_DATA) {
					disp = (dataset[tests][i] * (max[i] - min[i])) + min[i];
				}
				else {
					disp = dataset[tests][i];
				}
				
				acc += GetAcc(prediction[i].weight, disp);
			}
			disp = (double)prediction.size();
			acc = acc / disp;
			pCtr = pCtr / disp;
			cout << "Row Numeric Accuracy: " << acc << endl;
			cout << "Row Pass/Fail Accuracy: " << pCtr << endl;
			grAcc.push_back(pCtr);
			tpCtr += pCtr;
			totalAcc += acc;
		}
	}

	if (TESTED_DRAW > 0) {
		totalAcc = totalAcc / (double)TESTED_DRAW;
		tpCtr = tpCtr / (double)TESTED_DRAW;
		cout << endl << "Total Numeric Accuracy for " << TESTED_DRAW << " Tests: " << totalAcc << endl;
		cout << "Total Pass/Fail Accuracy: " << tpCtr << endl;
	}

	avPosMat = avPosMat / TESTED_DRAW;
	cout << "Average Positive Matches: " << avPosMat << endl;
	cout << "Fewest Positive Matches: " << minPosMat << endl;
	cout << "Most Positive Matches: " << maxPosMat << endl;
	cout << "Number of Up Fails: " << failUp << endl;
	cout << "Number of Down Fails: " << failDown << endl << endl;

	after = time(NULL);
	cout << endl << "Time taken in seconds: " << (after - before) << endl << endl;

	out = "";
	output.open("PassFailCtrs.csv");
	for (int i = 0; i < (int)grAcc.size(); i++) {
		out += to_string(grAcc[i]) + '\n';
	}
	output << out;
	output.close();

	system("pause");
	return 0;
}

int ListContains(const pattern &bar, vector<pattern> &p) {
	int result = -1; // returns -1 on not contained

	if (p.empty())
		return result;

	for (int i = 0; i < (int)p.size(); i++) {
		if (bar == p[i]) {
			result = i;
			break;
		}
	}

	return result;
}

bool UpperMatches(const pattern &a, const pattern &b) {
    int ctr = 0;

    if(a.width != b.width || a.height != b.height)
        return false;

    for(int i = 0; i < b.width; i++) {
        for(int j = 0; j < (b.height - 1); j++) {
            if(CompVals(a.image[j][i], b.image[j][i]))
                ctr++;
        }
    }

    return ctr == (a.width * (a.height - 1));
}

bool PatternMatches(const pattern& a, const pattern& b) {
	int ctr = 0;

	if (a.width != b.width || a.height != b.height)
		return false;


	//CompVals(a.image[j][i], b.image[j][i])
	for (int i = 0; i < b.width; i++) {
		for (int j = 0; j < b.height; j++) {
			if (CompVals(a.image[j][i], b.image[j][i]))
				ctr++;
		}
	}

	//double result = ctr / (double)(b.width * (b.height - 1));

	return ctr == (a.width * a.height);
}

bool CompVals(double a, double b) {
	//cout << a << " " << b << " | ";

	double error;

	if (a == 0 && b == 0)
		return true;

	error = abs(a - b);
	error = error / a;

	/*
	if (b != 0)
		equ = a / b;
	else if (a != 0)
		equ = b / a;

	if (abs(equ) > 1.000)
		equ = 1.0000 / equ;

	return equ >= ACCURACY;
	*/

	return abs(error) < (1 - ACCURACY);
}

double GetAcc(double a, double b) {

	double error;

	error = abs(a - b);
	error = error / a;

	/*
	if (b != 0)
	equ = a / b;
	else if (a != 0)
	equ = b / a;

	if (abs(equ) > 1.000)
	equ = 1.0000 / equ;

	return equ >= ACCURACY;
	*/

	return (1 - abs(error));
}

long long Sigmoid(long long x, long long mid, long long highest) {

	double L = (double)highest;

	double x0 = (double)mid;

	double k = 0.008;

	double exp = k * (x - x0);

	double val = L / (1.0 + (1.0 / pow(e, exp)));

	//rescale the value and cast back to int
	return (long long)val;
}

pattern PullFromDataset(int line, int offset, int width, int height, const vector<vector<double> > &data) {
	pattern result;

	result.width = width;
	result.height = height;



	for (int i = 0; i < width; i++) {
		for (int j = 0; j < height; j++) {

			if(line + j < (int)data.size())
				result.image[j][i] = data[line + j][offset + i];
			else
				result.image[j][i] = 0;

		}
	}

	return result;
}

bool IndicesMatch(const pattern &foo, const pattern &bar) {
	bool result = true;

	if (foo.width != bar.width)
		return false;

	for (int x = 0; x < foo.width; x++) {
		if (foo.indices[x] != bar.indices[x])
			return false;
	}

	return result;
}