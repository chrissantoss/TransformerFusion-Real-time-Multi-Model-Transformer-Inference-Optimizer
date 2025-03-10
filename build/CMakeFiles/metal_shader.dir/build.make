# CMAKE generated file: DO NOT EDIT!
# Generated by "Unix Makefiles" Generator, CMake Version 3.31

# Delete rule output on recipe failure.
.DELETE_ON_ERROR:

#=============================================================================
# Special targets provided by cmake.

# Disable implicit rules so canonical targets will work.
.SUFFIXES:

# Disable VCS-based implicit rules.
% : %,v

# Disable VCS-based implicit rules.
% : RCS/%

# Disable VCS-based implicit rules.
% : RCS/%,v

# Disable VCS-based implicit rules.
% : SCCS/s.%

# Disable VCS-based implicit rules.
% : s.%

.SUFFIXES: .hpux_make_needs_suffix_list

# Command-line flag to silence nested $(MAKE).
$(VERBOSE)MAKESILENT = -s

#Suppress display of executed commands.
$(VERBOSE).SILENT:

# A target that is always out of date.
cmake_force:
.PHONY : cmake_force

#=============================================================================
# Set environment variables for the build.

# The shell in which to execute make rules.
SHELL = /bin/sh

# The CMake executable.
CMAKE_COMMAND = /opt/homebrew/bin/cmake

# The command to remove a file.
RM = /opt/homebrew/bin/cmake -E rm -f

# Escaping for special characters.
EQUALS = =

# The top-level source directory on which CMake was run.
CMAKE_SOURCE_DIR = /Users/christophersantos/Desktop/Projects/intro_to_cuda

# The top-level build directory on which CMake was run.
CMAKE_BINARY_DIR = /Users/christophersantos/Desktop/Projects/intro_to_cuda/build

# Utility rule file for metal_shader.

# Include any custom commands dependencies for this target.
include CMakeFiles/metal_shader.dir/compiler_depend.make

# Include the progress variables for this target.
include CMakeFiles/metal_shader.dir/progress.make

CMakeFiles/metal_shader: transformer.metallib

transformer.metallib:
	@$(CMAKE_COMMAND) -E cmake_echo_color "--switch=$(COLOR)" --blue --bold --progress-dir=/Users/christophersantos/Desktop/Projects/intro_to_cuda/build/CMakeFiles --progress-num=$(CMAKE_PROGRESS_1) "Generating transformer.metallib"
	xcrun -sdk macosx metal -c /Users/christophersantos/Desktop/Projects/intro_to_cuda/src/transformer.metal -o /Users/christophersantos/Desktop/Projects/intro_to_cuda/build/transformer.air
	xcrun -sdk macosx metallib /Users/christophersantos/Desktop/Projects/intro_to_cuda/build/transformer.air -o /Users/christophersantos/Desktop/Projects/intro_to_cuda/build/transformer.metallib

CMakeFiles/metal_shader.dir/codegen:
.PHONY : CMakeFiles/metal_shader.dir/codegen

metal_shader: CMakeFiles/metal_shader
metal_shader: transformer.metallib
metal_shader: CMakeFiles/metal_shader.dir/build.make
.PHONY : metal_shader

# Rule to build all files generated by this target.
CMakeFiles/metal_shader.dir/build: metal_shader
.PHONY : CMakeFiles/metal_shader.dir/build

CMakeFiles/metal_shader.dir/clean:
	$(CMAKE_COMMAND) -P CMakeFiles/metal_shader.dir/cmake_clean.cmake
.PHONY : CMakeFiles/metal_shader.dir/clean

CMakeFiles/metal_shader.dir/depend:
	cd /Users/christophersantos/Desktop/Projects/intro_to_cuda/build && $(CMAKE_COMMAND) -E cmake_depends "Unix Makefiles" /Users/christophersantos/Desktop/Projects/intro_to_cuda /Users/christophersantos/Desktop/Projects/intro_to_cuda /Users/christophersantos/Desktop/Projects/intro_to_cuda/build /Users/christophersantos/Desktop/Projects/intro_to_cuda/build /Users/christophersantos/Desktop/Projects/intro_to_cuda/build/CMakeFiles/metal_shader.dir/DependInfo.cmake "--color=$(COLOR)"
.PHONY : CMakeFiles/metal_shader.dir/depend

