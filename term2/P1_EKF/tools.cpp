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

MatrixXd Tools::CalculateJacobian(const VectorXd& x_state) {
  /**
   * TODO:
   * Calculate a Jacobian here.
   */
	MatrixXd Hj(3, 4);

	// recover state parametersparameters
	float px = x_state(0);
	float py = x_state(1);
	float vx = x_state(2);
	float vy = x_state(3);

	// pre-compute a set of terms to avoid repeated calculation
	float c1 = px * px + py * py;
	float c2 = sqrt(c1);
	float c3 = (c1 * c2);

	// check division by zero
	if (fabs(c1) < 0.0001) {
		std::cout << "CalculateJacobian() - Error - Division by zero" << std::endl;
		return Hj;
	}

	// Compute the Jacobian matrx
	Hj << (px / c2), (py / c2), 0, 0,
		- (py / c1), (px / c1), 0, 0,
		py * (vx * py - vy * px) / c3, px * (vy * px - py * vx) / c3, px / c2, py / c2;

	return Hj;
}
