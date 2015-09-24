#pragma once
#pragma message("Compiling PCH - Should only happen once per project")

// std
#include <vector>
#include <iostream>
#include <fstream>
#include <iomanip>
#include <string>
#include <cmath>
#include <list>

// Boost.
//#define BOOST_LIB_DIAGNOSTIC
#include <boost/thread.hpp>
#include <boost/timer.hpp>
#include <boost/lexical_cast.hpp>
#include <boost/random/mersenne_twister.hpp>
#include <boost/random/linear_congruential.hpp>
//#include <boost/random/uniform_int.hpp>
#include <boost/random/uniform_real.hpp>
#include <boost/random/variate_generator.hpp>
