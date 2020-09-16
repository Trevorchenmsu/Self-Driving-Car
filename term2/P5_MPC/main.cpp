#include <math.h>
#include <uWS/uWS.h>
#include <chrono>
#include <iostream>
#include <thread>
#include <vector>
#include "Eigen-3.3/Eigen/Core"
#include "Eigen-3.3/Eigen/QR"
#include "MPC.h"
#include "json.hpp"
#include "helpers.h"

// for convenience
using json = nlohmann::json;
using Eigen::VectorXd;

// For converting back and forth between radians and degrees.
constexpr double pi() { return M_PI; }
double deg2rad(double x) { return x * pi() / 180; }
double rad2deg(double x) { return x * 180 / pi(); }

// convert from map coordinate to car coordinates
void map2car(double px, double py, double psi, const vector<double>& ptsx_map, const vector<double>& ptsy_map,
	VectorXd& ptsx_car, VectorXd& ptsy_car) {

	for (size_t i = 0; i < ptsx_map.size(); i++) {
		double dx = ptsx_map[i] - px;
		double dy = ptsy_map[i] - py;
		ptsx_car[i] = dx * cos(-psi) - dy * sin(-psi);
		ptsy_car[i] = dx * sin(-psi) + dy * cos(-psi);
	}
}

int main() {
  uWS::Hub h;

  // MPC is initialized here!
  MPC mpc;

  h.onMessage([&mpc](uWS::WebSocket<uWS::SERVER> ws, char *data, size_t length,
                     uWS::OpCode opCode) {
    // "42" at the start of the message means there's a websocket message event.
    // The 4 signifies a websocket message
    // The 2 signifies a websocket event
    string sdata = string(data).substr(0, length);
    //cout << sdata << endl;
    if (sdata.size() > 2 && sdata[0] == '4' && sdata[1] == '2') {
      string s = hasData(sdata);
      if (s != "") {
        auto j = json::parse(s);
        string event = j[0].get<string>();
        if (event == "telemetry") {
          // j[1] is the data JSON object
          vector<double> ptsx = j[1]["ptsx"]; 
          vector<double> ptsy = j[1]["ptsy"];
          double px = j[1]["x"];
          double py = j[1]["y"];
          double psi = j[1]["psi"];
          double v = j[1]["speed"];
		  double steer_value = j[1]["steering_angle"]; // steering angle is in the opposite direction
		  double throttle_value = j[1]["throttle"];

          // Calculate steering angle and throttle using MPC. Both are in between [-1, 1].

		  // convert from map coordinate to car coordinate
		  // Note: ptsx and ptsy are the coordinates in the map system
		  // ptsx_car and ptsy_car are the coordinates in the car system
		  // px and py are the reference point coordinates
		  VectorXd ptsx_car(ptsx.size());
		  VectorXd ptsy_car(ptsy.size());
		  map2car(px, py, psi, ptsx, ptsy, ptsx_car, ptsy_car);

		  // compute the coefficients
          auto coeffs = polyfit(ptsx_car, ptsy_car, 3);
               
		  // state in car coordniates {x, y, psi, v, cte, epsi}
          VectorXd state(6);

		  double cte = polyeval(coeffs, 0);  // px = 0, py = 0
		  double epsi = -atan(coeffs[1]);  // p    

          state << 0, 0, 0, v, cte, epsi;


		  // call MPC solver
          auto vars = mpc.Solve(state, coeffs);
          steer_value = vars[0];
          throttle_value = vars[1];

          json msgJson;
          // NOTE: Remember to divide by deg2rad(25) before you send the steering value back.
          // Otherwise the values will be in between [-deg2rad(25), deg2rad(25] instead of [-1, 1].
          msgJson["steering_angle"] = steer_value/(deg2rad(25));
          msgJson["throttle"] = throttle_value;

          //Display the MPC predicted trajectory 
		  // Points are in reference to the vehicle's coordinate system
		  // The points in the simulator are connected by a Green line
          vector<double> mpc_x_vals;
          vector<double> mpc_y_vals;
		  
		  for (size_t i = 2; i < vars.size(); i = i + 2) { //the first two are steer angle and throttle value
			  mpc_x_vals.push_back(vars[i]);
			  mpc_y_vals.push_back(vars[i + 1]);
		  }

          msgJson["mpc_x"] = mpc_x_vals;
          msgJson["mpc_y"] = mpc_y_vals;

          //Display the waypoints/reference line
		  // Points are in reference to the vehicle's coordinate system
		  // the points in the simulator are connected by a Yellow line
          vector<double> next_x_vals;
          vector<double> next_y_vals;
        
		  for (int i = 1; i < ptsx_car.size(); i++) { // index staring from 1 for visualize
												// only the reference point which is in the front of the car
			  next_x_vals.push_back(ptsx_car[i]);
			  next_y_vals.push_back(ptsy_car[i]);
		  }

          msgJson["next_x"] = next_x_vals;
          msgJson["next_y"] = next_y_vals;


          auto msg = "42[\"steer\"," + msgJson.dump() + "]";
          //std::cout << msg << std::endl;
          
          // Latency
          // The purpose is to mimic real driving conditions where
          // the car does actuate the commands instantly.
          //
          // Feel free to play around with this value but should be to drive
          // around the track with 100ms latency.
          //
          // NOTE: REMEMBER TO SET THIS TO 100 MILLISECONDS BEFORE
          // SUBMITTING.
          this_thread::sleep_for(chrono::milliseconds(100));
          ws.send(msg.data(), msg.length(), uWS::OpCode::TEXT);
        }
      } else {
        // Manual driving
        std::string msg = "42[\"manual\",{}]";
        ws.send(msg.data(), msg.length(), uWS::OpCode::TEXT);
      }
    }
  });

  // We don't need this since we're not using HTTP but if it's removed the
  // program
  // doesn't compile :-(
  h.onHttpRequest([](uWS::HttpResponse *res, uWS::HttpRequest req, char *data,
                     size_t, size_t) {
    const std::string s = "<h1>Hello world!</h1>";
    if (req.getUrl().valueLength == 1) {
      res->end(s.data(), s.length());
    } else {
      // i guess this should be done more gracefully?
      res->end(nullptr, 0);
    }
  });

  h.onConnection([&h](uWS::WebSocket<uWS::SERVER> ws, uWS::HttpRequest req) {
    std::cout << "Connected!!!" << std::endl;
  });

  h.onDisconnection([&h](uWS::WebSocket<uWS::SERVER> ws, int code,
                         char *message, size_t length) {
    ws.close();
    std::cout << "Disconnected" << std::endl;
  });

  int port = 4567;
  if (h.listen(port)) {
    std::cout << "Listening to port " << port << std::endl;
  } else {
    std::cerr << "Failed to listen to port" << std::endl;
    return -1;
  }
  h.run();
}
