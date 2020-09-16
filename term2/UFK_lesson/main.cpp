#include <iostream>
#include "Eigen/Dense"
#include <vector>
#include "ukf.h"

using Eigen::MatrixXd;
using Eigen::VectorXd;
using namespace std;
using std::vector;

int main() {

	//create a UKF instance
	UKF ukf;

	MatrixXd Xsig = MatrixXd(11, 5);
	ukf.GenerateSigmaPoints(&Xsig);

	MatrixXd Xsig_aug = MatrixXd(15, 7);
	ukf.AugmentedSigmaPoints(&Xsig_aug);

	MatrixXd Xsig_pred = MatrixXd(15, 5);
	ukf.SigmaPointPrediction(&Xsig_pred);

	VectorXd x_pred = VectorXd(5);
	MatrixXd P_pred = MatrixXd(5, 5);
	ukf.PredictMeanAndCovariance(&x_pred, &P_pred);

	VectorXd z_out = VectorXd(3);
	MatrixXd S_out = MatrixXd(3, 3);
	ukf.PredictRadarMeasurement(&z_out, &S_out);

	VectorXd x_out = VectorXd(5);
	MatrixXd P_out = MatrixXd(5, 5);
	ukf.UpdateState(&x_out, &P_out);

	// Print result
	//std::cout << "Xsig = " << std::endl << Xsig << std::endl;
	//std::cout << "Xsig_aug = " << std::endl << Xsig_aug << std::endl;
	//std::cout << "Xsig_pred = " << std::endl << Xsig_pred << std::endl;
	//std::cout << "x_pred = " << std::endl << x_pred << std::endl;
	//std::cout << "P_pred = " << std::endl << P_pred << std::endl;
	//std::cout << "z_out = " << std::endl << z_out << std::endl;
	//std::cout << "S_out = " << std::endl << S_out << std::endl;
	//std::cout << "x_out = " << std::endl << x_out << std::endl;
	//std::cout << "P_out = " << std::endl << P_out << std::endl;

	return 0;

}