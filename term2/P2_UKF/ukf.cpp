#include <iostream>
#include "ukf.h"
#include "Eigen/Dense"
#include <math.h>

using Eigen::MatrixXd;
using Eigen::VectorXd;
using namespace std;


 //Initializes Unscented Kalman filter
UKF::UKF() {

  // initially set to false, set to true in first call of ProcessMeasurement
  is_initialized_ = false;

  // time when the state is true, in us
  time_us_ = 0;

  // if this is false, laser measurements will be ignored (except during init)
  use_laser_ = true;

  // if this is false, radar measurements will be ignored (except during init)
  use_radar_ = true;

  // State dimension
  n_x_ = 5;

  // Augmented state dimension
  n_aug_ = 7;

  // Sigma point spreading parameter
  lambda_ = 3 - n_aug_;

  // Create augmented sigma point matrix
  Xsig_aug = MatrixXd(n_aug_, 2 * n_aug_ + 1);
  Xsig_aug.fill(0.0);

  // initial state vector
  x_ = VectorXd(n_x_);

  // initial covariance matrix
  P_ = MatrixXd(n_x_, n_x_);
  
  // predicted sigma points matrix
  Xsig_pred_ = MatrixXd(n_x_, 2 * n_aug_ + 1);

  //create vector for weights
  weights_ = VectorXd(2 * n_aug_ + 1);

  // Weights of sigma points
  weights_(0) = lambda_ / (lambda_ + n_aug_);
  weights_.tail(2 * n_aug_).fill(0.5 / (lambda_ + n_aug_));

  // Process noise standard deviation longitudinal acceleration in m/s^2
  std_a_ = 0.8;

  // Process noise standard deviation yaw acceleration in rad/s^2
  std_yawdd_ = 0.55;

  // Laser measurement noise standard deviation position1 in m
  std_laspx_ = 0.15;

  // Laser measurement noise standard deviation position2 in m
  std_laspy_ = 0.15;

  // Radar measurement noise standard deviation radius in m
  std_radr_ = 0.3;

  // Radar measurement noise standard deviation angle in rad
  std_radphi_ = 0.03;

  // Radar measurement noise standard deviation radius change in m/s
  std_radrd_ = 0.3;

  // The current NIS for radar
  NIS_radar_ = 0.0;

  // The current NIS for Lidar
  NIS_laser_ = 0.0;
  
}

UKF::~UKF() {}

void UKF::ProcessMeasurement(MeasurementPackage meas_package) {

  /*
  * Initialization
  */
  if (!is_initialized_) {
	  Initialization(meas_package);
  }

  //compute the time elapsed between the current and previous measurements
  float dt = (meas_package.timestamp_ - time_us_) / 1000000.0; //dt - expressed in seconds
  time_us_ = meas_package.timestamp_;

  /*
  * Prediction
  */
  Prediction(dt);

  /*
  * Update
  */
  if (meas_package.sensor_type_ == MeasurementPackage::RADAR) {
	  UpdateRadar(meas_package, dt);
  }
  else {
	  UpdateLidar(meas_package, dt);
  }

}

/************************* Initialization **************************************/

// Initialize the x_, time_us_, set is_initialized true after initialization
void UKF::Initialization(MeasurementPackage meas_package) {
	// First measurement
	x_ << 1, 1, 1, 1, 0.1;

	// First covariance matrix
	P_ << 1, 0, 0, 0, 0,
		0, 1, 0, 0, 0,
		0, 0, 1, 0, 0,
		0, 0, 0, 1, 0,
		0, 0, 0, 0, 1;

	if (meas_package.sensor_type_ == MeasurementPackage::RADAR) {
		// TODO: Convert radar from polar to cartesian coordinates 
		float ro = meas_package.raw_measurements_(0);
		float phi = meas_package.raw_measurements_(1);
		x_(0) = ro * cos(phi);
		x_(1) = ro * sin(phi);
	}
	else if (meas_package.sensor_type_ == MeasurementPackage::LASER && use_laser_) {
		x_(0) = meas_package.raw_measurements_(0);
		x_(1) = meas_package.raw_measurements_(1);
	}

	time_us_ = meas_package.timestamp_;

	// done initializing, no need to predict or update
	is_initialized_ = true;
	return;
}

/************************* Prediction ******************************************/

void UKF::Prediction(double delta_t) {
	// Calculate augmented sigma points: Xsig_aug_
	AugmentedSigmaPoints();

	// Predict the sigma points: Xsig_pred_
	SigmaPointPrediction(delta_t);

	// Calculate the predicted mean and covariance: x_ and P_
	PredictMeanAndCovariance();
}

// Augment Sigma Points outputs: Xsig_aug, x_aug
void UKF::AugmentedSigmaPoints() {
	/****************************************************************
	******************Augment Sigma Points***************************
	****************************************************************/
	// Create augmented mean vector
	VectorXd x_aug = VectorXd(n_aug_);
	x_aug.fill(0.0);

	// Create augmented state covariance;
	MatrixXd P_aug = MatrixXd(n_aug_, n_aug_);
	P_aug.fill(0.0);

	// Create the noise covariance matrix
	MatrixXd Q = MatrixXd(2, 2);
	Q << std_a_ * std_a_, 0,
		0, std_yawdd_* std_yawdd_;

	// Fill out the augmented state covariance matrix with x_, P_, Q
	x_aug.head(n_x_) = x_;
	P_aug.topLeftCorner(n_x_, n_x_) = P_;
	P_aug.bottomRightCorner(2, 2) = Q;

	// calculate square root of P_aug_sqrt
	MatrixXd P_aug_sqrt = P_aug.llt().matrixL();

	Xsig_aug.col(0) = x_aug;
	for (int i = 0; i < n_aug_; i++)
	{
		Xsig_aug.col(i + 1) = x_aug + sqrt(lambda_ + n_aug_) * P_aug_sqrt.col(i);
		Xsig_aug.col(i + 1 + n_aug_) = x_aug - sqrt(lambda_ + n_aug_) * P_aug_sqrt.col(i);
	}
}

// Sigma Point Prediction output: Xsig_pred_
void UKF::SigmaPointPrediction(double delta_t) {
	/****************************************************************
	 ******************Sigma Point Prediction************************
	 ***************************************************************/
	for (int i = 0; i < 2 * n_aug_ + 1; i++)
	{
		double px = Xsig_aug.col(i)[0];
		double py = Xsig_aug.col(i)[1];
		double velo = Xsig_aug.col(i)[2];
		double yaw = Xsig_aug.col(i)[3];
		double yaw_d = Xsig_aug.col(i)[4];
		double acc_long = Xsig_aug.col(i)[5];
		double acc_yaw = Xsig_aug.col(i)[6];

		if (fabs(yaw_d) > 0.001) {
			Xsig_pred_.col(i)[0] = px + velo / yaw_d * (sin(yaw + yaw_d * delta_t) - sin(yaw)) + 0.5 * delta_t * delta_t * cos(yaw) * acc_long;
			Xsig_pred_.col(i)[1] = py + velo / yaw_d * (-cos(yaw + yaw_d * delta_t) + cos(yaw)) + 0.5 * delta_t * delta_t * sin(yaw) * acc_long;
		}
		else {
			Xsig_pred_.col(i)[0] = px + velo * cos(yaw) * delta_t + 0.5 * delta_t * delta_t * cos(yaw) * acc_long;
			Xsig_pred_.col(i)[1] = py + velo * sin(yaw) * delta_t + 0.5 * delta_t * delta_t * sin(yaw) * acc_long;

		}

		Xsig_pred_.col(i)[2] = velo + delta_t * acc_long;
		Xsig_pred_.col(i)[3] = yaw + yaw_d * delta_t + 0.5 * delta_t * delta_t * acc_yaw;
		Xsig_pred_.col(i)[4] = yaw_d + delta_t * acc_yaw;
	}
}

// Predict Mean And Covariance Section outputs: prediced x_, P_
void UKF::PredictMeanAndCovariance() {
	/****************************************************************
	******************Predict Mean And Covariance********************
	****************************************************************/
	x_.fill(0.0);
	for (int i = 0; i < 2 * n_aug_ + 1; i++) {
		x_ += weights_(i) * Xsig_pred_.col(i);
	}

	P_.fill(0.0);
	for (int i = 0; i < 2 * n_aug_ + 1; i++) {
		VectorXd x_diff = Xsig_pred_.col(i) - x_;

		//angle normalized to [-pi, pi]
		while (x_diff(3) > M_PI) x_diff(3) -= 2. * M_PI;
		while (x_diff(3) < -M_PI) x_diff(3) += 2. * M_PI;

		P_ += weights_(i) * x_diff * x_diff.transpose();
	}
}


/************************* Update *********************************************/

// Update the Lidar measurement: x_, P_
void UKF::UpdateLidar(MeasurementPackage meas_package, double delta_t) {
  
	PredictLidarMeasurement();

	// Set the measurement dimension
	int n_z = 2;

	// Create vector for incoming radar measurement
	VectorXd z = VectorXd(n_z);
	z << meas_package.raw_measurements_(0), meas_package.raw_measurements_(1);

	// Create matrix for cross correlation Tc
	MatrixXd Tc = MatrixXd(n_x_, n_z);
	Tc.fill(0.0);

	for (int i = 0; i < 2 * n_aug_ + 1; i++) {
		Tc += weights_(i) * (Xsig_pred_.col(i) - x_) * (Zsig.col(i) - z_pred).transpose();
	}

	// create matrix for Kalman gain K
	MatrixXd K = MatrixXd(n_x_, n_z);
	K = Tc * S.inverse();

	// Update state
	x_ = x_ + K * (z - z_pred);

	// Update covariance matrix
	P_ = P_ - K * S * K.transpose();

	//calculate the radar NIS
	NIS_laser_ = (z - z_pred).transpose() * S.inverse() * (z - z_pred);
}

// Update the Radar measurement: x_, P_
void UKF::UpdateRadar(MeasurementPackage meas_package, double delta_t) {
  
	float ro = meas_package.raw_measurements_(0);
	float phi = meas_package.raw_measurements_(1);
	float ro_dot = meas_package.raw_measurements_(2);

	// Set the measurement dimension
	int n_z = 3;

	// Create vector for incoming radar measurement
	VectorXd z = VectorXd(n_z);
	z << ro, phi, ro_dot;

	PredictRadarMeasurement(); 

	// Create matrix for cross correlation Tc
	MatrixXd Tc = MatrixXd(n_x_, n_z);
	Tc.fill(0.0);

	for (int i = 0; i < 2 * n_aug_ + 1; i++) {
		//residual
		VectorXd z_diff = Zsig.col(i) - z_pred;

		//angle normalized to [-pi, pi]
		while (z_diff(1) > M_PI) z_diff(1) -= 2. * M_PI;
		while (z_diff(1) < -M_PI) z_diff(1) += 2. * M_PI;

		// state difference
		VectorXd x_diff = Xsig_pred_.col(i) - x_;

		//angle normalized to [-pi, pi]
		while (x_diff(3) > M_PI) x_diff(3) -= 2. * M_PI;
		while (x_diff(3) < -M_PI) x_diff(3) += 2. * M_PI;

		Tc += weights_(i) * x_diff * z_diff.transpose();
	}

	// create matrix for Kalman gain K
	MatrixXd K = Tc * S.inverse();

	//residual
	VectorXd z_diff = z - z_pred;

	//angle normalization
	while (z_diff(1) > M_PI) z_diff(1) -= 2. * 2 * M_PI;
	while (z_diff(1) < -M_PI) z_diff(1) += 2. * 2 * M_PI;

	// Update state and covariance matrix
	x_ = x_ + K * z_diff;
	P_ = P_ - K * S * K.transpose();

	//calculate the radar NIS
	NIS_radar_ = z_diff.transpose() * S.inverse() * z_diff;
}

// Predcit Radar Measurement outputs: z_pred, S, Zsig
void UKF::PredictRadarMeasurement() {
  /****************************************************************
  ******************Predict Radar Measurement*********************
  ****************************************************************/

  // Set measurement dimension, radar can measure rho, phi, rho_dot
  int n_z = 3;

  // create matrix for sigma points in measurement space
  Zsig = MatrixXd(n_z, 2 * n_aug_ + 1);

  // Create vector for mean predicted measurement
  z_pred = VectorXd(n_z);
  z_pred.fill(0.0);

  // Create measurement covariance matrix 
  S = MatrixXd(n_z, n_z);
  S.fill(0.0);

  // Create noise covariance matrix
  MatrixXd R = MatrixXd(n_z, n_z);
  R << std_radr_ * std_radr_, 0,						 0,
		0,					  std_radphi_ * std_radphi_, 0,
		0,					  0,						 std_radrd_ * std_radrd_;

  for (int i = 0; i < 2 * n_aug_ + 1; i++) {
	  double px = Xsig_pred_(0, i);
	  double py = Xsig_pred_(1, i);
	  double vel = Xsig_pred_(2, i);
	  double yaw = Xsig_pred_(3, i);

	  Zsig(0, i) = sqrt(px * px + py * py);
	  Zsig(1, i) = atan2(py, px);
	  Zsig(2, i) = (px * cos(yaw) * vel + py * sin(yaw) * vel) / Zsig(0, i);
	  z_pred += weights_(i) * Zsig.col(i);
  }

  for (int i = 0; i < 2 * n_aug_ + 1; i++) {
	  VectorXd z_diff = Zsig.col(i) - z_pred;

	  //angle normalized to [-pi, pi]
	  while (z_diff(1) > M_PI) z_diff(1) -= 2. * M_PI;
	  while (z_diff(1) < -M_PI) z_diff(1) += 2. * M_PI;

	  S += weights_(i) * z_diff * z_diff.transpose();
  }

  S += R;
}

// Predcit Lidar Measurement outputs: z_pred, S, Zsig
void UKF::PredictLidarMeasurement() {
  /****************************************************************
  ******************Predict Lidar Measurement*********************
  ****************************************************************/

  // Set measurement dimension, Lidar can measure px, py
  int n_z = 2;

  // create matrix for sigma points in measurement space
  Zsig = MatrixXd(n_z, 2 * n_aug_ + 1);

  // Create vector for mean predicted measurement
  z_pred = VectorXd(n_z);
  z_pred.fill(0.0);

  // Create measurement covariance matrix 
  S = MatrixXd(n_z, n_z);
  S.fill(0.0);

  // Create noise covariance matrix
  MatrixXd R = MatrixXd(n_z, n_z);
  R << std_laspx_ * std_laspx_, 0,
		0, std_laspy_ * std_laspy_;

  for (int i = 0; i < 2 * n_aug_ + 1; i++) {
	double px = Xsig_pred_(0, i);
	double py = Xsig_pred_(1, i);

	Zsig(0, i) = px;
	Zsig(1, i) = py;
	z_pred += weights_(i) * Zsig.col(i);
  }

  for (int i = 0; i < 2 * n_aug_ + 1; i++) {

	  VectorXd z_diff = Zsig.col(i) - z_pred;
	  S += weights_(i) * z_diff * z_diff.transpose();
  }

  S += R;
}

