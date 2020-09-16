#ifndef UKF_H_
#define UKF_H_

#include <iostream>
#include "Eigen/Dense"

using Eigen::MatrixXd;
using Eigen::VectorXd;

class UKF {

	public:
	/**
	 * Constructor.
	 */
	 UKF();

	/**
	* Destructor.
	*/
	virtual ~UKF();


	void GenerateSigmaPoints(MatrixXd* Xsig_out);
	void AugmentedSigmaPoints(MatrixXd* Xsig_out);
	void SigmaPointPrediction(MatrixXd* Xsig_out);
	void PredictMeanAndCovariance(VectorXd* x_out, MatrixXd* P_out);
	void PredictRadarMeasurement(VectorXd* z_out, MatrixXd* S_out);
	void UpdateState(VectorXd* x_out, MatrixXd* P_out);

};

#endif // UKF_H_