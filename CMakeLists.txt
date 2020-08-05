cmake_minimum_required(VERSION 3.0.2)
project(openvino_maskrcnn_inference)

## Compile as C++11, supported in ROS Kinetic and newer
add_compile_options(-std=c++11)

## Find catkin macros and libraries
## if COMPONENTS list like find_package(catkin REQUIRED COMPONENTS xyz)
## is used, also find other catkin packages


find_package(InferenceEngine)
if (NOT InferenceEngine_FOUND)
    message(FATAL_ERROR "")
endif()

set(_opencv_version 3)
find_package(OpenCV 3)
 
find_package(catkin REQUIRED COMPONENTS 
  cv_bridge
  image_transport
  roscpp
  sensor_msgs
  std_msgs
)

find_package(OpenCV ${_opencv_version} REQUIRED
  COMPONENTS
    opencv_core
    opencv_imgproc
    opencv_imgcodecs
  CONFIG
)


catkin_package(
  INCLUDE_DIRS include
  ${THIS_PACKAGE_INCLUDE_DIRS}
#  LIBRARIES openvino_maskrcnn_inference
#  CATKIN_DEPENDS cv_bridge image_transport roscpp sensor_msgs std_msgs
#  DEPENDS system_lib
)

 
## Specify additional locations of header files
## Your package locations should be listed before other locations
include_directories(
  include
  ${catkin_INCLUDE_DIRS}
  ${THIS_PACKAGE_INCLUDE_DIRS}
)

include_directories (
  include
  ${catkin_INCLUDE_DIRS}
  ${OpenCV_INCLUDE_DIRS}
  ${InferenceEngine_INCLUDE_DIRS}
  ${InferenceEngine_INCLUDE_DIRS}/../samples/build/thirdparty/gflags/include
  ${InferenceEngine_INCLUDE_DIRS}/../build/samples/thirdparty/gflags/include
  ${InferenceEngine_INCLUDE_DIRS}/../samples
  ${InferenceEngine_DIR}/../src
)

set(mask_SRCS 
src/main.cpp
)

add_executable(mask ${mask_SRCS})
target_link_libraries(mask ${catkin_LIBRARIES} ${InferenceEngine_LIBRARIES} ${IE_SAMPLE_DEPENDENCIES} ${OpenCV_LIBS}) 

 
