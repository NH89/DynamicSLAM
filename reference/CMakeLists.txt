# cmake needs this line
cmake_minimum_required(VERSION 2.8)

# Define project name
project(opencv_DynamicSLAM CXX)

# Find OpenCV, you may need to set OpenCV_DIR variable
# to the absolute path to the directory containing OpenCVConfig.cmake file
# via the command line or GUI
find_package(OpenCV REQUIRED)

# set flags for debugging
set(CMAKE_CXX_FLAGS "-ggdb -Wall -std=c++14 ") # changed from c++11
#/home/nick/Programming/otherLibraries/GSL

# If the package has been found, several variables will
# be set, you can find the full list with descriptions
# in the OpenCVConfig.cmake file.
# Print some message showing some of them
message(STATUS "OpenCV library status:")
message(STATUS "    version: ${OpenCV_VERSION}")
message(STATUS "    libraries: ${OpenCV_LIBS}")
message(STATUS "    include path: ${OpenCV_INCLUDE_DIRS}")

if(CMAKE_VERSION VERSION_LESS "2.8.11")
  # Add OpenCV headers location to your include paths
  include_directories(${OpenCV_INCLUDE_DIRS})
endif()

include_directories("$HOME/Programming/otherLibraries/GSL/include")

set(DynamicSLAM_0_5_files 
        DynamicSLAM_0_5.cpp
        image_column.cpp
        tracking_and_mapping.cpp 
        image_column.h
        tracking_and_mapping.h)        

# Declare the executable target built from your sources
add_executable(DynamicSLAM_05 ${DynamicSLAM_0_5_files})
#/home/nick/Programming/ComputerVision/DynamicSLAM/DynamicSLAM_0_3.cpp

# Link your application with OpenCV libraries
target_link_libraries(DynamicSLAM_05 ${OpenCV_LIBS})


####
#cmake_minimum_required(VERSION 2.6)
#find_package(OpenCV REQUIRED)
#project(tracker3D)
#set(tracker3D_files object_tracker.cpp tracker3D.cpp tracker_calibration.cpp tracker_calibration.h tracker3D.h object_tracker.h)
#add_executable(tracker3D ${tracker3D_files})
#target_link_libraries(tracker3D ${OpenCV_LIBS})
