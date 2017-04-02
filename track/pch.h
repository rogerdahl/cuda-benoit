#pragma once
#pragma message("Compiling PCH - Should only happen once per project")

// std
#include <cmath>
#include <fstream>
#include <iomanip>
#include <iostream>
#include <list>
#include <string>
#include <vector>

// Boost.
//#define BOOST_LIB_DIAGNOSTIC
#include <boost/lexical_cast.hpp>
#include <boost/random/linear_congruential.hpp>
#include <boost/random/mersenne_twister.hpp>
#include <boost/thread.hpp>
#include <boost/timer.hpp>
//#include <boost/random/uniform_int.hpp>
#include <boost/random/uniform_real.hpp>
#include <boost/random/variate_generator.hpp>
