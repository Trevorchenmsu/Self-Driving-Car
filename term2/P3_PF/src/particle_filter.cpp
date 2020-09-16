/**
 * particle_filter.cpp
 *
 */

#include "particle_filter.h"
#include "helper_functions.h"

#include <math.h>
#include <algorithm>
#include <iostream>
#include <iterator>
#include <numeric>
#include <random>
#include <string>
#include <vector>

using std::string;
using std::vector;
using namespace std;

// declare a random engine to be used across multiple and various method calls
static default_random_engine gen;

/*
 * init Initializes particle filter by initializing particles to 
 *  Gaussian distribution around first position and all the weights to 1.
 */
void ParticleFilter::init(double x, double y, double theta, double std[]) {

  // Set the number of particles
  num_particles = 100;

  // Initialize x, y, theta with random Gaussian noise
  normal_distribution<double> N_x(x, std[0]);
  normal_distribution<double> N_y(y, std[1]);
  normal_distribution<double> N_theta(theta, std[2]);

  // resize the vectors of particles
  particles.resize(num_particles);

  // Create the particles
  for (auto& particle : particles) {
	  particle.x = N_x(gen);
	  particle.y = N_y(gen);
	  particle.theta = N_theta(gen);
	  particle.weight = 1.0;
  }

  is_initialized = true;

}

/*
 * prediction Predicts the state for the next time step using the process model.
 */
void ParticleFilter::prediction(double delta_t, double std_pos[], 
                                double velocity, double yaw_rate) {
	
	// generate random Gaussian noise
	normal_distribution<double> N_x(0, std_pos[0]);
	normal_distribution<double> N_y(0, std_pos[1]);
	normal_distribution<double> N_theta(0, std_pos[2]);

	for (auto& particle : particles) {

		// add measurements to each particle
		if (fabs(yaw_rate) < 0.0001) {  // constant velocity
			particle.x += velocity * delta_t * cos(particle.theta);
			particle.y += velocity * delta_t * sin(particle.theta);

		}
		else {
			particle.x += velocity / yaw_rate * (sin(particle.theta + yaw_rate * delta_t) - sin(particle.theta));
			particle.y += velocity / yaw_rate * (cos(particle.theta) - cos(particle.theta + yaw_rate * delta_t));
			particle.theta += yaw_rate * delta_t;
		}
		// predicted particles with added sensor noise
		particle.x += N_x(gen);
		particle.y += N_y(gen);
		particle.theta += N_theta(gen);
	}
}

/**
 * dataAssociation Finds which observations correspond to which 
 *  landmarks using a nearest-neighbors data association
 */
void ParticleFilter::dataAssociation(vector<LandmarkObs> predicted, 
                                     vector<LandmarkObs>& observations) {

	for (unsigned int i = 0; i < observations.size(); i++) {

		// Initialize minimum distance to maximum possible
		double min_dist = numeric_limits<double>::max();

		for (unsigned int j = 0; j < predicted.size(); j++) {
			
			// Calcute the distance between predicted measurement and observed measurement
			double cur_dist = dist(predicted[j].x, predicted[j].y, observations[i].x, observations[i].y);

			if (min_dist > cur_dist) {
				min_dist = cur_dist;
				observations[i].id = predicted[j].id;
			}
		}
	}
}


/**
  * updateWeights Updates the weights for each particle based on the likelihood of the observed measurements.
  * Steps:
  * 1: Collect the landmarks within the sensor range for each particle, as predictions
  * 2: Convert the observations from vehicle coordinate to map coordinate
  * 3: Use dataAssociation(predictions, observations) to find the landmark index for each observation
  * 4: Update the weights of each particle using a multi-variate Gaussian distribution
  * Note: Observations in the VEHICLE'S coordinate system. Particles in the MAP'S coordinate system
  */
void ParticleFilter::updateWeights(double sensor_range, double std_landmark[], 
                                   const vector<LandmarkObs> observations, 
                                   const Map map_landmarks) {

	for (auto& particle : particles) {
	
		// Step 1: collect the landmarks within the sensor range for each particle
		vector<LandmarkObs> predictions;
		for (const auto& lm : map_landmarks.landmark_list) {
			double distance = dist(particle.x, particle.y, lm.x_f, lm.y_f);
			if (distance < sensor_range) { 
				predictions.push_back(LandmarkObs{ lm.id_i, lm.x_f, lm.y_f });
			}
		}

		// Step 2: convert the observations from car coordinate to map coordinate
		vector<LandmarkObs> obs_car2map;
		double cos_theta = cos(particle.theta);
		double sin_theta = sin(particle.theta);
		
		for (const auto& obs : observations) {
			LandmarkObs tmp;
			tmp.x = obs.x * cos_theta - obs.y * sin_theta + particle.x;
			tmp.y = obs.x * sin_theta + obs.y * cos_theta + particle.y;
			obs_car2map.push_back(tmp);
		}

		// Step 3: Use dataAssociation(predictions, observations) to find the landmark index for each observation
		dataAssociation(predictions, obs_car2map);

		particle.weight = 1.0;
		// Step 4: Update the weights of each particle using a multi-variate Gaussian distribution
		for (const auto& obs : obs_car2map) {

			// Method from Junsheng Fu
			/*Map::single_landmark_s landmark = map_landmarks.landmark_list.at(obs.id - 1);
			double x_term = pow(obs.x - landmark.x_f, 2) / (2 * pow(std_landmark[0], 2));
			double y_term = pow(obs.y - landmark.y_f, 2) / (2 * pow(std_landmark[1], 2));*/
			
			// Method from Jeremy

			int associated_prediction = obs.id;
			double pr_x = 0.0;
			double pr_y = 0.0;

			// get the x,y coordinates of the prediction associated with the current observation
			for (const auto& pred : predictions) {
				if (pred.id == associated_prediction) {
					pr_x = pred.x;
					pr_y = pred.y;
				}
			}
			double x_term = pow(obs.x - pr_x, 2) / (2 * pow(std_landmark[0], 2));
			double y_term = pow(obs.y - pr_y, 2) / (2 * pow(std_landmark[1], 2));

			double w = exp(-(x_term + y_term)) / (2 * M_PI * std_landmark[0] * std_landmark[1]);
			particle.weight *= w;
		}

		weights.push_back(particle.weight);
	}
}

/**
  * resample Resamples from the updated set of particles to form
  * the new set of particles.
  */
void ParticleFilter::resample() {

	//vector<Particle> particles_new;
	////particles_new.resize(num_particles);

	//// get all of the current weights
	//vector<double> weights;
	//for (int i = 0; i < num_particles; i++) {
	//	weights.push_back(particles[i].weight);
	//}

	//// generate random starting index for resampling wheel
	//uniform_int_distribution<int> uniintdist(0, num_particles - 1);
	//auto index = uniintdist(gen);

	//double max_w = *max_element(weights.begin(), weights.end());

	//// uniform random distribution [0.0, max_w)
	//uniform_real_distribution<double> unirealdist(0.0, max_w);

	//double beta = 0.0;
	//for (int i = 0; i < num_particles; i++) {
	//	beta += unirealdist(gen) * 2.0;
	//	while (beta > weights[index]) {
	//		beta -= weights[index];
	//		index = (index + 1) % num_particles;
	//	}
	//	particles_new.push_back(particles[index]);
	//}

	//particles = particles_new;


	// generate distribution according to weights
	random_device rd;
	mt19937 gen(rd());
	discrete_distribution<> distr(weights.begin(), weights.end());

	// create resampled particles
	vector<Particle> resampled_particles;
	resampled_particles.resize(num_particles);

	// resample the particles according to weights
	for (int i = 0; i < num_particles; i++) {
		int idx = distr(gen);
		resampled_particles[i] = particles[idx];
	}

	// assign the resampled_particles to the previous particles
	particles = resampled_particles;

	// clear the weight vector for the next round
	weights.clear();
}


void ParticleFilter::SetAssociations(Particle& particle, 
                                     const vector<int>& associations, 
                                     const vector<double>& sense_x, 
                                     const vector<double>& sense_y) {
  // particle: the particle to which assign each listed association, 
  //   and association's (x,y) world coordinates mapping
  // associations: The landmark id that goes along with each listed association
  // sense_x: the associations x mapping already converted to world coordinates
  // sense_y: the associations y mapping already converted to world coordinates
  particle.associations= associations;
  particle.sense_x = sense_x;
  particle.sense_y = sense_y;
}

string ParticleFilter::getAssociations(Particle best) {
  vector<int> v = best.associations;
  std::stringstream ss;
  copy(v.begin(), v.end(), std::ostream_iterator<int>(ss, " "));
  string s = ss.str();
  s = s.substr(0, s.length()-1);  // get rid of the trailing space
  return s;
}

string ParticleFilter::getSenseCoord(Particle best, string coord) {
  vector<double> v;

  if (coord == "X") {
    v = best.sense_x;
  } else {
    v = best.sense_y;
  }

  std::stringstream ss;
  copy(v.begin(), v.end(), std::ostream_iterator<float>(ss, " "));
  string s = ss.str();
  s = s.substr(0, s.length()-1);  // get rid of the trailing space
  return s;
}