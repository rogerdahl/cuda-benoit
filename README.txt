PLAYER MAKEFILE WORKS UNDER LINUX NOW.

EDITOR MAKEFILE DOES NOT WORK YET.


apt-get install libglew-dev
sudo apt-get install libboost-serialization-dev
sudo apt-get install wx2.8-headers libwxgtk2.8-0 libwxgtk2.8-dev
sudo ln -sv /usr/include/wx-2.8/wx/ /usr/include/wx

nvcc -o benoit -arch=sm_20 -I/usr/include/GL/ -I/usr/local/cuda-7.5/samples/common/inc/ -I../../lib -I../../player/player/ ../../editor/track/track.cpp *.cpp *.cu -lboost_program_options -lboost_system -lboost_filesystem -lboost_serialization -lGLEW -lglut -lGL

For Kepler and newer, turn on the single precision mode.
