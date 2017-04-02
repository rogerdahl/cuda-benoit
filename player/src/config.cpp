#include "pch.h"

#include "config.h"

// stl
#include <fstream>
#include <iostream>
#include <string>
using namespace std;

// boost::program_options
#include <boost/program_options/options_description.hpp>
#include <boost/program_options/parsers.hpp>
#include <boost/program_options/variables_map.hpp>
namespace po = boost::program_options;

const string cfg_file_name("player.cfg");

bool read_config(Configuration& cfg)
{
  // Declare a group of options that will be allowed both on command line and in
  // config file
  po::options_description desc("Configuration");
  desc.add_options()(
      "device", po::value<s32>(&cfg.device_)->default_value(-1), "CUDA device")(
      "vsync", po::value<bool>(&cfg.vsync_)->default_value(true),
      "Vertical sync")(
      "vsync_interval", po::value<u32>(&cfg.vsync_interval_)->default_value(1),
      "Vertical sync interval")(
      "fullscreen", po::value<bool>(&cfg.fullscreen_)->default_value(true),
      "Fullscreen")(
      "screen_w", po::value<u32>(&cfg.screen_w_)->default_value(1920),
      "Screen resolution, width")(
      "screen_h", po::value<u32>(&cfg.screen_h_)->default_value(1080),
      "Screen resolution, height")(
      "transform_ss_x", po::value<u32>(&cfg.transform_ss_x_)->default_value(1),
      "Supersampling, horizontal")(
      "transform_ss_y", po::value<u32>(&cfg.transform_ss_y_)->default_value(1),
      "Supersampling, vertical")(
      "zoom_step", po::value<double>(&cfg.zoom_step_)->default_value(1.005),
      "Zoom step")(
      "fractal_box_ss", po::value<u32>(&cfg.fractal_box_ss_)->default_value(1),
      "Fractal box supersampling")(
      "boxes_per_frame",
      po::value<u32>(&cfg.boxes_per_frame_)->default_value(1),
      "Fractal boxes per frame")(
      "bailout", po::value<u32>(&cfg.bailout_)->default_value(2000),
      "Mandelbrot bailout value")(
      "grayscale", po::value<bool>(&cfg.grayscale_)->default_value(false),
      "Grayscale")(
      "timers", po::value<bool>(&cfg.timers_)->default_value(false),
      "Display timer bars")(
      "single_precision",
      po::value<bool>(&cfg.single_precision_)->default_value(false),
      "Single precision fractal calculation");

  ifstream cfg_file;
  cfg_file.open(cfg_file_name.c_str());
  if (!cfg_file.good()) {
    cout << "Could not open configuration file: " << cfg_file_name << endl;
    return false;
  }

  po::variables_map vm;
  try {
    po::store(po::parse_config_file(cfg_file, desc, false), vm);
  } catch (std::exception& e) {
    cout << "Error in configuration file: " << cfg_file_name << endl;
    cout << e.what() << endl;
    return false;
  }
  po::notify(vm);

  // if (vm.count("compression")) {
  //  cout << "Compression level was set to "
  //    << vm["compression"].as<int>() << ".\n";
  //} else {
  //  cout << "Compression level was not set.\n";
  //}

  return true;
}
