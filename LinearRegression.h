#include <float.h>
#include <math.h>
#include <random>
#include <string>
#include <vector>

/**
*   @author Bruce Blum
*
*   Implementation of linear regression using gradient descent.
*   Basic usage:
*   Construct a object with LinearRegression().
*   Load training data with loadTrainingData().
*   Load corresponding target data with loadTargetData().
*   Set the learning rate with setLearningRate().
*   Perform gradient descent with gradientDescent().
*/
class LinearRegression {
    public:
        LinearRegression();

        /**
         * Calculates the mean squared error of the training data.
         * 
         * @returns Cost MSE on the current traning data using the current weights.
         */
        double cost();

        /**
         *  Helper function that converts a vector of doubles to a string. 
         */
        static std::string dblVectToString(const std::vector<double>& dVec);

        /**
         * Computes the hypothesis minus the corresponding target value.
         * 
         * @param trainingVecIndex row index of the training data vector to 
         *          compute the differece of the hypothesis using the current
         *          weights and the given target value in the target vector.
         */
        double differenceEquation(unsigned long trainingVecIndex);

        /**
         * Perform a estimate using the current weights.
         * 
         * @param testVect A vector of test data similar to a row of training
         *          data to get an estimate of the target value.
         */
        double estimate(std::vector<double> testVec);

        /**
         * @returns The current cost based on the last calculation using the cost()
         *      function.
         */
        double getCost();

        double getLearningRate();

        std::vector<double> getTargetData();

        std::vector<std::vector<double>> getTrainingData();

        std::vector<double> getWeights();

        /**
         * Performs linear regression on the training and target data with the goal
         * of updating the weights that results in a useful estimate of test data.
         * 
         * @param maxSteps Number of steps to take during the update process.
         */
        void gradientDecent(unsigned int maxSteps = 100);

        /**
         * Puts the target data into the target data vector.
         * 
         * @param target Row vector that is equal in lenght to the number of
         *      rows in the training data vector.
         */
        void loadTargetData(std::vector<double> target);

        /**
         * Puts the training data along with an additional row of "1.0" from the
         * equation of hypothesis = w0x0 + w1x1 + ... + wnxn + 1.0xn+1.
         * The training data vector size is increased by 1.
         */
        void loadTrainingData(std::vector<std::vector<double>> trainingSetVec);
    
        /**
         * Set the weights to specific values. Useful to start from a know set.
         * 
         * @param weights Row vector that is equal in length plus 1 of the number
         *      of features in the training data. 
         */
        void reloadWeights(std::vector<double> weights);
        
        /**
         * @param rate The learning rate that the weights will be updated by for each
         *      step.
         */
        void setLearningRate(double rate);
        
        /**
         * Performs gradient descent on all the weights.
         */
        void updateWeights();

    private:
        double learningRate{0.000001};
        double currentCost{0.0};
        std::vector<double> targetVec;
        std::vector<std::vector<double>> trainingVec;
        std::vector<double> weightVec;
};