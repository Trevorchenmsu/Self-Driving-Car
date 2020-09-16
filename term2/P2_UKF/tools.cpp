#include "tools.h"
#include <iostream>

using Eigen::VectorXd;
using Eigen::MatrixXd;
using std::vector;

Tools::Tools() {}

Tools::~Tools() {}

VectorXd Tools::CalculateRMSE(const vector<VectorXd> &estimations,
                              const vector<VectorXd> &ground_truth) {
  /**
   * TODO: Calculate the RMSE here.
   */

	VectorXd RMSE(4);
	RMSE << 0, 0, 0, 0;

	// Check the validity of the following inputs:
	// The estimation vector size should not be zero;
	// The estimation vector size should equal to ground truth vector size.
	if (estimations.size() != ground_truth.size() || estimations.size() == 0) {
		std::cout << "Invalid estimation or ground_truth data" << std::endl;
		return RMSE;
	}

	// Accumulate the squared residuals
	for (unsigned int i = 0; i < estimations.size(); i++) {
		VectorXd residual = estimations[i] - ground_truth[i];

		// Coefficient-wise multiplication
		residual = residual.array() * residual.array();
		RMSE += residual;

	}

	// Calculate the mean
	RMSE = RMSE / estimations.size();
	// Calculate the square root
	RMSE = RMSE.array().sqrt();

	return RMSE;
}