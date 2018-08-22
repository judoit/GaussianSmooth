cmake_minimum_required(VERSION 3.11)


project(untitled2)

set(CMAKE_CXX_FLAGS "${CMAKE_CXX_FLAGS} -std=c++11")

find_package( OpenCV REQUIRED )
set(SOURCE_FILES main.cpp)
add_executable(untitled2 ${SOURCE_FILES})
include_directories(${OpenCV_INCLUDE_DIRS})
target_link_libraries( untitled2 ${OpenCV_LIBS} )