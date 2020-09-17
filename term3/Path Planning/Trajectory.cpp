/*
 * Trajectory.cpp
 */

#include "Trajectory.h"
#include "Constants.h"
#include <iostream>


Trajectory::Trajectory(){
}

Trajectory::Trajectory(vector<double> x,
		vector <double> y,
		vector <double> s,
		vector <double> d,
		double target_speed,
		int target_lane,
		int final_lane) {
	this->x = x;
	this->y = y;
	this->s = s;
	this->d = d;
	this->target_speed = target_speed;
	this->target_lane = target_lane;
	this->final_lane = final_lane;
}

Trajectory::~Trajectory() {
}


const vector<double>& Trajectory::getX() const {
	return x;
}

const vector<double>& Trajectory::getY() const {
	return y;
}

const vector<double>& Trajectory::getD() const {
	return d;
}

const vector<double>& Trajectory::getS() const {
	return s;
}

double Trajectory::getTargetSpeed() const {
	return target_speed;
}

int Trajectory::getTargetLane() const {
	return target_lane;
}

int Trajectory::getFinalLane() const {
	return final_lane;
}

int Trajectory::get_lane(double car_d) {
	if (car_d < LANE_WIDTH) {
		return 0;  // left lane
	}
	else {
		if (car_d < 2 * LANE_WIDTH) {
			return 1;  // middle lane
		}
		else {
			return 2;  // right lane
		}
	}
}

void Trajectory::debug_info() {
	cout << "Target speed=" << this->target_speed
		<< " target lane " << this->target_lane
		<< " final lane " << this->final_lane
		<< "\n";
}


