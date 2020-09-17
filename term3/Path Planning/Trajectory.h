/*
 * Trajectory.h
 */

#ifndef SRC_TRAJECTORY_H_
#define SRC_TRAJECTORY_H_

#include <vector>

using namespace std;

class Trajectory {
public:
	Trajectory();

	Trajectory(vector<double> x,
			vector <double> y,
			vector <double> s,
			vector <double> d,
			double target_speed,
			int target_lane,
			int final_lane);

	virtual ~Trajectory();

	
	const vector<double>& getX() const;
	const vector<double>& getY() const;
	const vector<double>& getS() const;
	const vector<double>& getD() const;
	double getTargetSpeed() const;
	int getTargetLane() const;
	int getFinalLane() const;
	
	static int get_lane(double car_d);
	void debug_info();

private:
	vector<double> x;
	vector<double> y;
	vector<double> s;
	vector<double> d;
	double target_speed;
	int target_lane;
	int final_lane;
};

#endif /* SRC_TRAJECTORY_H_ */
