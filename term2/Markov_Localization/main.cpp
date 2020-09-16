#include <iostream>
#include <vector>
#include <fstream>
#include <sstream>
#include <string>
#include "measurement_package.h"
#include "map.h"
#include "help_functions.h"
#include "bayesian_filter.h"

using namespace std;
int main() {
	/******************************************************************************
	   *  declaration:
	   *****************************************************************************/

	   //define example: 01, 02, 03, 04
	string example_string = "01";

	//declare map:
	Map map_1d;

	//declare measurement list:
	std::vector<MeasurementPackage> measurement_pack_list;

	//declare helpers:
	HelpFunctions helper;

	//define file names:
	char in_file_name_ctr[1024];
	char in_file_name_obs[1024];
	char in_file_name_gt[1024];

	/******************************************************************************
	 *  read map and measurements:                           *
	 *****************************************************************************/
	 //read map:
	helper.ReadMapData("data/map_1d.txt", map_1d);

	//define file name of controls:
	sprintf_s(in_file_name_ctr, "data/example%s/control_data.txt",
		example_string.c_str());

	//define file name of observations:
	sprintf_s(in_file_name_obs, "data/example%s/observations/",
		example_string.c_str());

	//read in data to measurement package list:
	helper.ReadMeasurementData(in_file_name_ctr,
		in_file_name_obs,
		measurement_pack_list);

	/*for (int i = 0; i < map_1d.landmark_list_.size(); i++) {
		std::cout << "ID: "<< map_1d.landmark_list_[i].id << "\t"
				  <<"value in x: "<< map_1d.landmark_list_[i].x<< std::endl;
	}

	for (int i = 0; i < measurement_pack_list.size(); i++) {
		std::cout <<"Step "<<i<<" includes the move"
				  << measurement_pack_list[i].control_.delta_x 
				  <<" [m] in driving direction" << std::endl;
		if (measurement_pack_list[i].observation_.distances.size() < 1) {
			std::cout << " No observations in step " << i << std::endl;
		}
		else {
			std::cout<<"Number of observations in current step:"
				     << measurement_pack_list[i].observation_.distances.size() 
					 << std::endl;

			for (int j = 0; j < measurement_pack_list[i].observation_.distances.size(); j++) {
				std::cout << " Distance to a landmark:"
					<< measurement_pack_list[i].observation_.distances[j]
					<< "m" << std::endl;
			}
		}	
	}

	return 0;*/



	/*******************************************************************************
	 *  start 1d_bayesian filter                           *
	 *******************************************************************************/

	 //create instance of 1d_bayesian localization filter:
	BayesianFilter localization_1d_bayesian;

	//define number of time steps:
	size_t T = measurement_pack_list.size();

	//cycle:
	for (size_t t = 0; t < T; ++t) {

		//Call 1d_bayesian filter:
		localization_1d_bayesian.ProcessMeasurement(measurement_pack_list[t],
			map_1d,
			helper);
	}

	/*******************************************************************************
	 *  print/compare results:                           *
	 ********************************************************************************/
	 //define file name of gt data:
	sprintf_s(in_file_name_gt, "data/example%s/gt_example%s.txt", example_string.c_str(), example_string.c_str());

	///compare gt data with results:
	helper.CompareData(in_file_name_gt, localization_1d_bayesian.belief_x);

	return 0;
}