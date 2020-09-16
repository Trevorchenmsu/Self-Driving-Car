#include <iostream>
#include "Eigen/Dense"
#include "ukf.h"
#include <math.h>
#define _USE_MATH_DEFINES

using Eigen::MatrixXd;
using Eigen::VectorXd;
using namespace std;

UKF::UKF() {}

UKF:: ~UKF() {}

void UKF::GenerateSigmaPoints(MatrixXd* Xsig_out) {

	//set state dimension
	int n_x = 5;

	//define spreading parameter
	double lambda = 3 - n_x;

	// set example state
	VectorXd x = VectorXd(n_x);
	x << 5.7441,
		 1.3800,
		 2.2049,
		 0.5015,
		 0.3528;

	// set example covariance matrix
	MatrixXd P = MatrixXd(n_x, n_x);
	P << 0.0043, -0.0013, 0.0030, -0.0022, -0.0020,
		-0.0013,  0.0077, 0.0011,  0.0071,  0.0060,
		 0.0030,  0.0011, 0.0054,  0.0007,  0.0008,
		-0.0022,  0.0071, 0.0007,  0.0098,  0.0100,
		-0.0020,  0.0060, 0.0008,  0.0100,  0.0123;
	// create sigma point matrix
	MatrixXd Xsig = MatrixXd(n_x, 2 * n_x + 1);

	// calculate square root of P
	MatrixXd A = P.llt().matrixL();

	/*******************************************************
	 * Student part begin
	********************************************************/
	// your code goes here

	// set sigma points as columns of matrix Xsig
	Xsig.col(0) = x;
	for (int i = 0; i < n_x; i++)
	{
		Xsig.col(i + 1) = x + sqrt(lambda + n_x) * A.col(i);
		Xsig.col(i + 1 + n_x) = x- sqrt(lambda + n_x) * A.col(i);
	}

	/*******************************************************
	 * Student part end
	********************************************************/
	//print result
	//std::cout << "Xsig = " << std::endl << Xsig << std::endl;

	// write result
	*Xsig_out = Xsig;

}

void UKF::AugmentedSigmaPoints(MatrixXd* Xsig_outAug) {
	// Set state dimension
	int n_x = 5;

	// Set augmented dimension
	int n_aug = 7;

	// Process noise standard deviation longitudinal acceleration in m/s^2
	double sta_a = 0.2;

	// Process noise standard deviation yaw acceleration in rad/s^2
	double std_yawdd = 0.2;

	//define spreading parameter
	double lambda = 3 - n_aug;

	// set example state
	VectorXd x = VectorXd(n_x);
	x << 5.7441,
		 1.3800,
		 2.2049,
		 0.5015,
		 0.3528;

	// set examole covariance matrix
	MatrixXd P = MatrixXd(n_x, n_x);
	P << 0.0043, -0.0013, 0.0030, -0.0022, -0.0020,
		-0.0013,  0.0077, 0.0011,  0.0071,  0.0060,
		 0.0030,  0.0011, 0.0054,  0.0007,  0.0008,
		-0.0022,  0.0071, 0.0007,  0.0098,  0.0100,
		-0.0020,  0.0060, 0.0008,  0.0100,  0.0123;
	
	// Create augmented mean vector
	VectorXd x_aug = VectorXd(7);
	x_aug << 0, 0, 0, 0, 0, 0, 0;

	// Create augmented state covariance;
	MatrixXd P_aug = MatrixXd(7, 7);
	P_aug.fill(0.0);

	// Create sigma point matrix
	MatrixXd Xsig_aug = MatrixXd(n_aug, 2 * n_aug + 1);

	// Create the noise covariance matrix
	MatrixXd Q = MatrixXd(2, 2);
	Q << 0.2 * 0.2, 0,
		 0, 0.2* 0.2;

	x_aug.head(5) = x;
	P_aug.topLeftCorner(n_x, n_x) = P;
	P_aug.bottomRightCorner(2, 2) = Q;

	// calculate square root of P_aug
	MatrixXd A = P_aug.llt().matrixL();

	/*******************************************************
	 * Student part begin
	********************************************************/
	// your code goes here

	// calculate sigma points
	Xsig_aug.col(0) = x_aug;
	for (int i = 0; i < n_aug; i++)
	{
		Xsig_aug.col(i + 1) = x_aug + sqrt(lambda + n_aug) * A.col(i);
		Xsig_aug.col(i + 1 + n_aug) = x_aug - sqrt(lambda + n_aug) * A.col(i);
	}
	// set sigma points as columns of matrix Xsig

	// create augmented mean state
	// create augmented covariance matrix
	// create square root matrix
	// create augmented sigma points

/*******************************************************
 * Student part end
********************************************************/
//print result
	//std::cout << "Xsig_aug = " << std::endl << x_aug << std::endl;
	//std::cout << "P_aug = " << std::endl << Xsig_aug << std::endl;
	// write result
	*Xsig_outAug = x_aug;
}

void UKF::SigmaPointPrediction(MatrixXd* Xsig_out) {
	// Set state dimension
	int n_x = 5;

	// Set augmented dimension
	int n_aug = 7;

	// Create example sigma point matrix
	MatrixXd Xsig_aug = MatrixXd(n_aug, 2 * n_aug + 1);
		Xsig_aug<<
		5.7441,  5.85768,   5.7441,   5.7441,   5.7441,   5.7441,  5.7441,  5.7441,  5.63052,   5.7441,   5.7441,   5.7441,   5.7441,   5.7441,   5.7441,
		  1.38,  1.34566,  1.52806,     1.38,     1.38,     1.38,    1.38,    1.38,  1.41434,  1.23194,     1.38,     1.38,     1.38,     1.38,     1.38,
		2.2049,  2.28414,  2.24557,  2.29582,   2.2049,   2.2049,  2.2049,  2.2049,  2.12566,  2.16423,  2.11398,   2.2049,   2.2049,   2.2049,   2.2049,
		0.5015,  0.44339, 0.631886, 0.516923, 0.595227,   0.5015,  0.5015,  0.5015,  0.55961, 0.371114, 0.486077, 0.407773,   0.5015,   0.5015,   0.5015,
		0.3528, 0.299973, 0.462123, 0.376339,  0.48417, 0.418721,  0.3528,  0.3528, 0.405627, 0.243477, 0.329261,  0.22143, 0.286879,   0.3528,   0.3528,
		     0,        0,        0,        0,        0,        0, 0.34641,       0,        0,        0,        0,        0,        0, -0.34641,        0,
		     0,        0,        0,        0,        0,        0,       0, 0.34641,        0,        0,        0,        0,        0,        0, -0.34641;

		// create matrix with predicted sigma points as columns
		MatrixXd Xsig_pred = MatrixXd(n_x, 2 * n_aug + 1);

		double delta_t = 0.1; // time diff in sec

		/***********************************************************************
		 * Student part begin
		***********************************************************************/
		// Predict sigma points
		// Avoid division by zero
		// Write predicted sigma points into right column
		

		for (int i = 0; i < 2 * n_aug + 1; i++)
		{
			double velo = Xsig_aug.col(i)[2];
			double phi = Xsig_aug.col(i)[3];
			double phi_dot = Xsig_aug.col(i)[4];
			double acc_long = Xsig_aug.col(i)[5];
			double acc_yaw = Xsig_aug.col(i)[6];

			if (phi_dot != 0) {
				Xsig_pred.col(i)[0] = Xsig_aug.col(i)[0] + velo / phi_dot * (sin(phi + phi_dot * delta_t) - sin(phi)) + 0.5 * delta_t* delta_t *cos(phi)* acc_long;
				Xsig_pred.col(i)[1] = Xsig_aug.col(i)[1] + velo / phi_dot * (-cos(phi + phi_dot * delta_t) + cos(phi)) + 0.5 * delta_t * delta_t * sin(phi) * acc_long;
				Xsig_pred.col(i)[2] = Xsig_aug.col(i)[2] + delta_t * acc_long;
				Xsig_pred.col(i)[3] = Xsig_aug.col(i)[3] + phi_dot * delta_t + 0.5* delta_t * delta_t* acc_yaw;
				Xsig_pred.col(i)[4] = Xsig_aug.col(i)[4] + delta_t* acc_yaw;
			}
			else {
				Xsig_pred.col(i)[0] = Xsig_aug.col(i)[0] + velo * cos(phi) * delta_t + 0.5* delta_t * delta_t*cos(phi)* acc_long;
				Xsig_pred.col(i)[1] = Xsig_aug.col(i)[1] + velo * sin(phi) * delta_t + 0.5 * delta_t * delta_t * sin(phi) * acc_long;
				Xsig_pred.col(i)[2] = Xsig_aug.col(i)[2] + delta_t * acc_long;
				Xsig_pred.col(i)[3] = Xsig_aug.col(i)[3] + phi_dot * delta_t+0.5* delta_t*delta_t* acc_yaw;
				Xsig_pred.col(i)[4] = Xsig_aug.col(i)[4] + delta_t* acc_yaw;
			
			}

		}

		// write result
		*Xsig_out = Xsig_pred;
}

void UKF::PredictMeanAndCovariance(VectorXd* x_out, MatrixXd* P_out) {
	// Set state dimension
	int n_x = 5;

	// Set augmented dimension
	int n_aug = 7;

	// Set the number of sigma points
	int n_aug_sigma = 2 * n_aug + 1;

	// Define spreading parameter
	double lambda = 3 - n_aug;

	// Create example matrix with predicted sigma points
	MatrixXd Xsig_pred = MatrixXd(n_x, n_aug_sigma);
	Xsig_pred <<
		5.93553,  6.06251,  5.92217,   5.9415,  5.92361,  5.93516, 5.93705,  5.93553,  5.80832,  5.94481,  5.92935,  5.94553,  5.93589, 5.93401,  5.93553,
		1.48939,  1.44673,  1.66484,  1.49719,    1.508,  1.49001, 1.49022,  1.48939,   1.5308,  1.31287,  1.48182,  1.46967,  1.48876, 1.48855,  1.48939,
		 2.2049,  2.28414,  2.24557,  2.29582,   2.2049,   2.2049, 2.23954,   2.2049,  2.12566,  2.16423,  2.11398,   2.2049,   2.2049, 2.17026,   2.2049,
		0.53678, 0.473387, 0.678098, 0.554557, 0.643644, 0.543372, 0.53678, 0.538512, 0.600173, 0.395462, 0.519003, 0.429916, 0.530188, 0.53678, 0.535048,
		 0.3528, 0.299973, 0.462123, 0.376339,  0.48417, 0.418721,  0.3528, 0.387441, 0.405627, 0.243477, 0.329261,  0.22143, 0.286879,  0.3528, 0.318159;

	// Create vector for predicted state
	VectorXd x = VectorXd(n_x);
	x.fill(0.0);

	// Create covariance matrix for prediction
	MatrixXd P = MatrixXd(n_x, n_x);
	P.fill(0.0);

	// Create weights of sigma points
	VectorXd weights = VectorXd(n_aug_sigma);
	weights(0) = lambda / (lambda + n_aug);
	weights.tail(n_aug_sigma - 1).fill(0.5 / (lambda + n_aug));

	for (int i = 0; i < n_aug_sigma; i++) {		
		x += weights(i) * Xsig_pred.col(i);
	}

	for (int i = 0; i < n_aug_sigma; i++) {	
		VectorXd x_diff = Xsig_pred.col(i) - x;

		//angle normalized to [-pi, pi]
		while (x_diff(3) > M_PI) x_diff(3) -= 2. * M_PI;
		while (x_diff(3) < -M_PI) x_diff(3) += 2. * M_PI;

		P += weights(i) * x_diff * x_diff.transpose();
	}

	// write result
	*x_out = x;
	*P_out = P;
}

void UKF::PredictRadarMeasurement(VectorXd* z_out, MatrixXd* S_out) {
	// Set state dimension
	int n_x = 5;

	// Set augmented dimension
	int n_aug = 7;

	// Set the number of sigma points
	int n_aug_sigma = 2 * n_aug + 1;

	// Set measurement dimension, radar can measure rho, phi, rho_dot
	int n_z = 3;

	// Define spreading parameter
	double lambda = 3 - n_aug;

	// Create weights of sigma points
	VectorXd weights = VectorXd(n_aug_sigma);
	weights(0) = lambda / (lambda + n_aug);
	weights.tail(n_aug_sigma - 1).fill(0.5 / (lambda + n_aug));

	// radar measurement noise standard deviation radius in m
	double std_radR = 0.3;

	// radar measurement noise standard deviation angle in rad
	double std_radPhi = 0.3;

	// radar measurement noise standard deviation radius change in m/s
	double std_radRd = 0.1;

	// create example matrix with predicted sigma points
	MatrixXd Xsig_pred = MatrixXd(n_x, 2 * n_aug + 1);
	Xsig_pred<<
			5.93553,  6.06251,  5.92217,   5.9415,  5.92361,  5.93516, 5.93705,  5.93553,  5.80832,  5.94481,  5.92935,  5.94553,  5.93589, 5.93401,  5.93553,
			1.48939,  1.44673,  1.66484,  1.49719,    1.508,  1.49001, 1.49022,  1.48939,   1.5308,  1.31287,  1.48182,  1.46967,  1.48876, 1.48855,  1.48939,
			 2.2049,  2.28414,  2.24557,  2.29582,   2.2049,   2.2049, 2.23954,   2.2049,  2.12566,  2.16423,  2.11398,   2.2049,   2.2049, 2.17026,   2.2049,
			0.53678, 0.473387, 0.678098, 0.554557, 0.643644, 0.543372, 0.53678, 0.538512, 0.600173, 0.395462, 0.519003, 0.429916, 0.530188, 0.53678, 0.535048,
			 0.3528, 0.299973, 0.462123, 0.376339,  0.48417, 0.418721,  0.3528, 0.387441, 0.405627, 0.243477, 0.329261,  0.22143, 0.286879,  0.3528, 0.318159;

	// create matrix for sigma points in measurement space
	MatrixXd Zsig = MatrixXd(n_z, n_aug_sigma);
	Zsig.fill(0.0);

	// Create vector for mean predicted measurement
	VectorXd z_pred = VectorXd(n_z);
	z_pred.fill(0.0);

	// Create measurement covariance matrix 
	MatrixXd S = MatrixXd(n_z, n_z);
	S.fill(0.0);

	// create noise covariance matrix
	MatrixXd R = MatrixXd(n_z, n_z);
	R << std_radR * std_radR,					   0,					 0,
						   0, std_radPhi* std_radPhi,					 0,
						   0,					   0, std_radRd* std_radRd;

	for (int i = 0; i < n_aug_sigma; i++) {
	/*	double px = Xsig_pred(0, i);
		double py = Xsig_pred(1, i);
		double vel = Xsig_pred(2, i);
		double phi = Xsig_pred(3, i);*/

		// The second method to represent the elements in a matrix
		double px = Xsig_pred.col(i)(0);
		double py = Xsig_pred.col(i)(1);
		double vel = Xsig_pred.col(i)(2);
		double phi = Xsig_pred.col(i)(3);

		Zsig(0, i) = sqrt(px * px + py * py);
		Zsig(1, i) = atan2(py, px);
		Zsig(2, i) = (px * cos(phi) * vel + py * sin(phi) * vel) / Zsig(0, i);
		z_pred += weights(i) * Zsig.col(i);
	}


	for (int i = 0; i < n_aug_sigma; i++) {
		VectorXd z_diff = Zsig.col(i) - z_pred;

		//angle normalized to [-pi, pi]
		while (z_diff(1) > M_PI) z_diff(1) -= 2. * M_PI;
		while (z_diff(1) < -M_PI) z_diff(1) += 2. * M_PI;

		S += weights(i) * z_diff * z_diff.transpose() ;
	}

	S = S + R;

	// write result
	*z_out = z_pred;
	*S_out = S;
}

void UKF::UpdateState(VectorXd* x_out, MatrixXd* P_out) {

	// Set state dimension
	int n_x = 5;

	// Set augmented dimension
	int n_aug = 7;

	// Set the number of sigma points
	int n_aug_sigma = 2 * n_aug + 1;

	// Set measurement dimension, radar can measure rho, phi, rho_dot
	int n_z = 3;

	// Define spreading parameter
	double lambda = 3 - n_aug;

	// Create weights of sigma points
	VectorXd weights = VectorXd(n_aug_sigma);
	weights(0) = lambda / (lambda + n_aug);
	weights.tail(n_aug_sigma - 1).fill(0.5 / (lambda + n_aug));

	// create example matrix with predicted sigma points
	MatrixXd Xsig_pred = MatrixXd(n_x, 2 * n_aug + 1);
	Xsig_pred <<
		5.93553,  6.06251,  5.92217,   5.9415,  5.92361,  5.93516, 5.93705,  5.93553,  5.80832,  5.94481,  5.92935,  5.94553,  5.93589, 5.93401,  5.93553,
		1.48939,  1.44673,  1.66484,  1.49719,    1.508,  1.49001, 1.49022,  1.48939,   1.5308,  1.31287,  1.48182,  1.46967,  1.48876, 1.48855,  1.48939,
		 2.2049,  2.28414,  2.24557,  2.29582,   2.2049,   2.2049, 2.23954,   2.2049,  2.12566,  2.16423,  2.11398,   2.2049,   2.2049, 2.17026,   2.2049,
		0.53678, 0.473387, 0.678098, 0.554557, 0.643644, 0.543372, 0.53678, 0.538512, 0.600173, 0.395462, 0.519003, 0.429916, 0.530188, 0.53678, 0.535048,
		 0.3528, 0.299973, 0.462123, 0.376339,  0.48417, 0.418721,  0.3528, 0.387441, 0.405627, 0.243477, 0.329261,  0.22143, 0.286879,  0.3528, 0.318159;

	// Create example vector for predicted state mean
	VectorXd x = VectorXd(n_x);
	x <<
		5.93445,
		1.48885,
		 2.2049,
		0.53678,
		 0.3528;
	
	// Create example matrix for predicted state vovariance
	MatrixXd P = MatrixXd(n_x, n_x);
	P <<
		  0.0054808, -0.00249899,  0.00340521,  -0.0035741, -0.00309082,
		-0.00249899,   0.0110551,  0.00151803,  0.00990779,  0.00806653,
		 0.00340521,  0.00151803,   0.0057998, 0.000780142, 0.000800107,
		 -0.0035741,  0.00990779, 0.000780142,   0.0119239,     0.01125,
		-0.00309082,  0.00806653, 0.000800107,     0.01125,      0.0127;

	// Create example matrix with sigma points in measurement space
	MatrixXd Zsig = MatrixXd(n_z, n_aug_sigma);
	Zsig <<
		 6.11954,  6.23274,  6.15173,  6.12723,  6.11255,  6.11933,  6.12122,  6.11954,  6.00666,  6.08805,  6.11171,  6.12448,  6.11974,  6.11786,  6.11954,
		0.245852, 0.234254, 0.274047, 0.246849, 0.249279, 0.245965, 0.245923, 0.245852, 0.257693, 0.217354, 0.244897, 0.242331, 0.245738, 0.245779, 0.245852,
		 2.11225,  2.21914,  2.06475,  2.18799,  2.03565,   2.1081,  2.14548,  2.11115,  2.00221,  2.12999,  2.03506,  2.16622,   2.1163,  2.07902,  2.11334;

	// Create example vector for predicted measurement mean
	VectorXd z_pred = VectorXd(n_z);
	z_pred <<
		 6.11934,
		0.245833,
		 2.10274;

	// Create example matrix for predicted measurement covariance
	MatrixXd S = MatrixXd(n_z, n_z);
	S <<
		   0.0946306, -0.000145108,   0.00408754,
		-0.000145108,     0.090318, -0.000781354,
		  0.00408754, -0.000781354,    0.0180469;

	// Create example vector for incoming radar measurement
	VectorXd z = VectorXd(n_z);
	z <<
		5.9214,		//rho in m
		0.2187,		//phi in rad
		2.0062;		//rho_dot in m/s

	// Create matrix for cross correlation Tc
	MatrixXd Tc = MatrixXd(n_x, n_z);
	Tc.fill(0.0);

	for (int i = 0; i < n_aug_sigma; i++) {
		VectorXd z_diff = Zsig.col(i) - z_pred;
		VectorXd x_diff = Xsig_pred.col(i) - x;

		//angle normalized to [-pi, pi]
		while (z_diff(1) > M_PI) z_diff(1) -= 2. * M_PI;
		while (z_diff(1) < -M_PI) z_diff(1) += 2. * M_PI;

		while (x_diff(3) > M_PI) x_diff(3) -= 2. * M_PI;
		while (x_diff(3) < -M_PI) x_diff(3) += 2. * M_PI;

		Tc += weights(i) * x_diff * z_diff.transpose();
	}
	
	// create matrix for Kalman gain K
	MatrixXd K = MatrixXd(n_x, n_z);
	K = Tc * S.inverse();

	//residual
	VectorXd z_diff = z - z_pred;

	//angle normalization
	while (z_diff(1) > M_PI) z_diff(1) -= 2. * 2 * M_PI;
	while (z_diff(1) < -M_PI) z_diff(1) += 2. * 2 * M_PI;

	// Update state
	x = x + K * z_diff;

	// Update covariance matrix
	P = P - K * S * K.transpose();
	
	// write result
	*x_out = x;
	*P_out = P;
}