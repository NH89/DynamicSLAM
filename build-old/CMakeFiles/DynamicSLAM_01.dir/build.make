# CMAKE generated file: DO NOT EDIT!
# Generated by "Unix Makefiles" Generator, CMake Version 3.5

# Delete rule output on recipe failure.
.DELETE_ON_ERROR:


#=============================================================================
# Special targets provided by cmake.

# Disable implicit rules so canonical targets will work.
.SUFFIXES:


# Remove some rules from gmake that .SUFFIXES does not remove.
SUFFIXES =

.SUFFIXES: .hpux_make_needs_suffix_list


# Suppress display of executed commands.
$(VERBOSE).SILENT:


# A target that is always out of date.
cmake_force:

.PHONY : cmake_force

#=============================================================================
# Set environment variables for the build.

# The shell in which to execute make rules.
SHELL = /bin/sh

# The CMake executable.
CMAKE_COMMAND = /usr/bin/cmake

# The command to remove a file.
RM = /usr/bin/cmake -E remove -f

# Escaping for special characters.
EQUALS = =

# The top-level source directory on which CMake was run.
CMAKE_SOURCE_DIR = /home/nick/Programming/ComputerVision/DynamicSLAM

# The top-level build directory on which CMake was run.
CMAKE_BINARY_DIR = /home/nick/Programming/ComputerVision/DynamicSLAM/build

# Include any dependencies generated for this target.
include CMakeFiles/DynamicSLAM_01.dir/depend.make

# Include the progress variables for this target.
include CMakeFiles/DynamicSLAM_01.dir/progress.make

# Include the compile flags for this target's objects.
include CMakeFiles/DynamicSLAM_01.dir/flags.make

CMakeFiles/DynamicSLAM_01.dir/DynamicSLAM.cpp.o: CMakeFiles/DynamicSLAM_01.dir/flags.make
CMakeFiles/DynamicSLAM_01.dir/DynamicSLAM.cpp.o: ../DynamicSLAM.cpp
	@$(CMAKE_COMMAND) -E cmake_echo_color --switch=$(COLOR) --green --progress-dir=/home/nick/Programming/ComputerVision/DynamicSLAM/build/CMakeFiles --progress-num=$(CMAKE_PROGRESS_1) "Building CXX object CMakeFiles/DynamicSLAM_01.dir/DynamicSLAM.cpp.o"
	/usr/bin/c++   $(CXX_DEFINES) $(CXX_INCLUDES) $(CXX_FLAGS) -o CMakeFiles/DynamicSLAM_01.dir/DynamicSLAM.cpp.o -c /home/nick/Programming/ComputerVision/DynamicSLAM/DynamicSLAM.cpp

CMakeFiles/DynamicSLAM_01.dir/DynamicSLAM.cpp.i: cmake_force
	@$(CMAKE_COMMAND) -E cmake_echo_color --switch=$(COLOR) --green "Preprocessing CXX source to CMakeFiles/DynamicSLAM_01.dir/DynamicSLAM.cpp.i"
	/usr/bin/c++  $(CXX_DEFINES) $(CXX_INCLUDES) $(CXX_FLAGS) -E /home/nick/Programming/ComputerVision/DynamicSLAM/DynamicSLAM.cpp > CMakeFiles/DynamicSLAM_01.dir/DynamicSLAM.cpp.i

CMakeFiles/DynamicSLAM_01.dir/DynamicSLAM.cpp.s: cmake_force
	@$(CMAKE_COMMAND) -E cmake_echo_color --switch=$(COLOR) --green "Compiling CXX source to assembly CMakeFiles/DynamicSLAM_01.dir/DynamicSLAM.cpp.s"
	/usr/bin/c++  $(CXX_DEFINES) $(CXX_INCLUDES) $(CXX_FLAGS) -S /home/nick/Programming/ComputerVision/DynamicSLAM/DynamicSLAM.cpp -o CMakeFiles/DynamicSLAM_01.dir/DynamicSLAM.cpp.s

CMakeFiles/DynamicSLAM_01.dir/DynamicSLAM.cpp.o.requires:

.PHONY : CMakeFiles/DynamicSLAM_01.dir/DynamicSLAM.cpp.o.requires

CMakeFiles/DynamicSLAM_01.dir/DynamicSLAM.cpp.o.provides: CMakeFiles/DynamicSLAM_01.dir/DynamicSLAM.cpp.o.requires
	$(MAKE) -f CMakeFiles/DynamicSLAM_01.dir/build.make CMakeFiles/DynamicSLAM_01.dir/DynamicSLAM.cpp.o.provides.build
.PHONY : CMakeFiles/DynamicSLAM_01.dir/DynamicSLAM.cpp.o.provides

CMakeFiles/DynamicSLAM_01.dir/DynamicSLAM.cpp.o.provides.build: CMakeFiles/DynamicSLAM_01.dir/DynamicSLAM.cpp.o


# Object files for target DynamicSLAM_01
DynamicSLAM_01_OBJECTS = \
"CMakeFiles/DynamicSLAM_01.dir/DynamicSLAM.cpp.o"

# External object files for target DynamicSLAM_01
DynamicSLAM_01_EXTERNAL_OBJECTS =

DynamicSLAM_01: CMakeFiles/DynamicSLAM_01.dir/DynamicSLAM.cpp.o
DynamicSLAM_01: CMakeFiles/DynamicSLAM_01.dir/build.make
DynamicSLAM_01: /usr/local/lib/libopencv_cudabgsegm.so.3.1.0
DynamicSLAM_01: /usr/local/lib/libopencv_cudaobjdetect.so.3.1.0
DynamicSLAM_01: /usr/local/lib/libopencv_cudastereo.so.3.1.0
DynamicSLAM_01: /usr/local/lib/libopencv_shape.so.3.1.0
DynamicSLAM_01: /usr/local/lib/libopencv_stitching.so.3.1.0
DynamicSLAM_01: /usr/local/lib/libopencv_superres.so.3.1.0
DynamicSLAM_01: /usr/local/lib/libopencv_videostab.so.3.1.0
DynamicSLAM_01: /usr/local/lib/libopencv_viz.so.3.1.0
DynamicSLAM_01: /usr/local/lib/libopencv_cudafeatures2d.so.3.1.0
DynamicSLAM_01: /usr/local/lib/libopencv_objdetect.so.3.1.0
DynamicSLAM_01: /usr/local/lib/libopencv_cudacodec.so.3.1.0
DynamicSLAM_01: /usr/local/lib/libopencv_calib3d.so.3.1.0
DynamicSLAM_01: /usr/local/lib/libopencv_cudaoptflow.so.3.1.0
DynamicSLAM_01: /usr/local/lib/libopencv_cudawarping.so.3.1.0
DynamicSLAM_01: /usr/local/lib/libopencv_features2d.so.3.1.0
DynamicSLAM_01: /usr/local/lib/libopencv_flann.so.3.1.0
DynamicSLAM_01: /usr/local/lib/libopencv_highgui.so.3.1.0
DynamicSLAM_01: /usr/local/lib/libopencv_ml.so.3.1.0
DynamicSLAM_01: /usr/local/lib/libopencv_photo.so.3.1.0
DynamicSLAM_01: /usr/local/lib/libopencv_cudaimgproc.so.3.1.0
DynamicSLAM_01: /usr/local/lib/libopencv_cudafilters.so.3.1.0
DynamicSLAM_01: /usr/local/lib/libopencv_cudaarithm.so.3.1.0
DynamicSLAM_01: /usr/local/lib/libopencv_video.so.3.1.0
DynamicSLAM_01: /usr/local/lib/libopencv_videoio.so.3.1.0
DynamicSLAM_01: /usr/local/lib/libopencv_imgcodecs.so.3.1.0
DynamicSLAM_01: /usr/local/lib/libopencv_imgproc.so.3.1.0
DynamicSLAM_01: /usr/local/lib/libopencv_core.so.3.1.0
DynamicSLAM_01: /usr/local/lib/libopencv_cudev.so.3.1.0
DynamicSLAM_01: CMakeFiles/DynamicSLAM_01.dir/link.txt
	@$(CMAKE_COMMAND) -E cmake_echo_color --switch=$(COLOR) --green --bold --progress-dir=/home/nick/Programming/ComputerVision/DynamicSLAM/build/CMakeFiles --progress-num=$(CMAKE_PROGRESS_2) "Linking CXX executable DynamicSLAM_01"
	$(CMAKE_COMMAND) -E cmake_link_script CMakeFiles/DynamicSLAM_01.dir/link.txt --verbose=$(VERBOSE)

# Rule to build all files generated by this target.
CMakeFiles/DynamicSLAM_01.dir/build: DynamicSLAM_01

.PHONY : CMakeFiles/DynamicSLAM_01.dir/build

CMakeFiles/DynamicSLAM_01.dir/requires: CMakeFiles/DynamicSLAM_01.dir/DynamicSLAM.cpp.o.requires

.PHONY : CMakeFiles/DynamicSLAM_01.dir/requires

CMakeFiles/DynamicSLAM_01.dir/clean:
	$(CMAKE_COMMAND) -P CMakeFiles/DynamicSLAM_01.dir/cmake_clean.cmake
.PHONY : CMakeFiles/DynamicSLAM_01.dir/clean

CMakeFiles/DynamicSLAM_01.dir/depend:
	cd /home/nick/Programming/ComputerVision/DynamicSLAM/build && $(CMAKE_COMMAND) -E cmake_depends "Unix Makefiles" /home/nick/Programming/ComputerVision/DynamicSLAM /home/nick/Programming/ComputerVision/DynamicSLAM /home/nick/Programming/ComputerVision/DynamicSLAM/build /home/nick/Programming/ComputerVision/DynamicSLAM/build /home/nick/Programming/ComputerVision/DynamicSLAM/build/CMakeFiles/DynamicSLAM_01.dir/DependInfo.cmake --color=$(COLOR)
.PHONY : CMakeFiles/DynamicSLAM_01.dir/depend

