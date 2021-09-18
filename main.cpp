#include "LinearRegression.h"
#include <iostream>
#include <fstream>

using namespace std;

/**
*   Example of how to use the LinearRegression class.
*   Program requires two data files for the training data and
*   target data. For example, house price estimate based on bedrooms,
*   bathrooms, and square feet. The target data file rows would be the
*   sale price for the corresponding row in the training data file.
*   Training Data File:
*   3   2   1100

*   Target Data File:
*   250000
*   @author Bruce Blum 
*/
int main(){
    string trainDataFName;
    string targetDataFName;
    unsigned int miniBatchSize{100};    //Number of samples to use per weight update.
    unsigned int numOfIterations{100};   //Number of updates to weights.
    
    vector<vector<double>> trainingSetVec;  //Holds training data.
    vector<double> targetVec;   //Holds target data.

    vector<double> testVect;
    int bedrooms{0};
    int baths{0};
    int squareFeet{0};
    int price{0};

    cout << "Enter training data file name or T(rainingData.txt): ";
    cin >> trainDataFName;
    if(trainDataFName.compare("T") == 0){
        trainDataFName = "TrainingData.txt";
    }

    ifstream insTrain{trainDataFName};
    if(!insTrain){ 
        cout << "Not able to open data file " << trainDataFName;
        return 0;
    }

    cout << "Enter target data file name or T(argetData.txt): ";
    cin >> targetDataFName;
    if(targetDataFName.compare("T") == 0){
        targetDataFName = "TargetData.txt";
    }

    ifstream insTarget{targetDataFName};
    if(!insTarget){
        cout << "Not able to open data file " << targetDataFName;
        return 0;
    }

    cout << endl;

    // Read in the training data file and put each row into a vector.
    trainingSetVec.clear();

    while(!insTrain.eof()) {
        vector<double> dataRow;
        
        if(insTrain >> bedrooms >> baths >> squareFeet) {
            dataRow.push_back(bedrooms);
            dataRow.push_back(baths);
            dataRow.push_back(squareFeet);
            cout << LinearRegression::dblVectToString(dataRow) << endl;
            trainingSetVec.push_back(dataRow);
        } else {
            break;
        }
    }

    // Read in the target data file and put it in a vector.
    targetVec.clear();

    while(!insTarget.eof()){
        if(insTarget >> price) {
            targetVec.push_back(price);
        } else{
            break;
        }
    }
    
    cout << LinearRegression::dblVectToString(targetVec);

    cout << "Training set size " << trainingSetVec.size() << endl;
    cout << "Target set size " << targetVec.size() << endl;
    if(trainingSetVec.size() != targetVec.size()){
        cout << "Training set size is not the same a target set size." << endl;
    } else {
        cout << "Training and target data set loaded." << endl;
    }

    // Create the LinearRegression class.
    LinearRegression* linReg = new LinearRegression();
    
    // Load the training data into the LinearRegression object.
    linReg->loadTrainingData(trainingSetVec);

    cout << "Size of loaded training data: " << linReg->getTrainingData().size() << endl;
    for(unsigned int i = 0; i < linReg->getTrainingData().size(); i++) {
        vector<double> trainRow = linReg->getTrainingData().at(i);
        cout << LinearRegression::dblVectToString(trainRow) << endl;
    }

    // Load the target data into the LinearRegression object.
    linReg->loadTargetData(targetVec);

    cout << "Loaded target data: " << endl;

    for(unsigned int i = 0; i < linReg->getTargetData().size(); i++) {
        cout << to_string(linReg->getTargetData().at(i)) << endl;
    }

    cout << "Enter number of iterations to perform for each weight update: ";
    cin >> numOfIterations;

    // Perform gradien decent on the data.
    linReg->gradientDecent(numOfIterations);

    cout << "Weights: " << endl;
    for(int i = 0; i < linReg->getWeights().size(); i++){
        cout << i << ": " << linReg->getWeights().at(i) << " ";
    }
  
    cout << endl;
    
    // Prompt for estimation of sale price for a house.
    while(true){
        testVect.clear();
        cout << "Enter bedrooms: ";
        cin >> bedrooms;
        cout << "Enter baths: ";
        cin >> baths;
        cout << "Enter square feet: ";
        cin >> squareFeet;

        testVect.push_back(bedrooms);
        testVect.push_back(baths);
        testVect.push_back(squareFeet);

        price = linReg->estimate(testVect);

        cout << "Estimated price " << price << endl;
    }

    return 0;
}