cmake_minimum_required(VERSION 3.10)

project(FeatureMatch)

set(CMAKE_BUILD_TYPE Debug)
set(CMAKE_CXX_FLAGS_RELEASE "-O3 -fPIC")
set(CMAKE_MODULE_PATH ${CMAKE_MODULE_PATH} "${CMAKE_SOURCE_DIR}/cmake")

find_package(OpenCV REQUIRED)
find_package(Eigen 3.0 REQUIRED)
message(STATUS "opencv version: ${OpenCV_VERSION}")

include_directories(
        ${OpenCV_INCLUDE_DIRS}
)

add_executable(FeatureMatch feature.cpp)

target_link_libraries(FeatureMatch PRIVATE
        ${OpenCV_LIBS}
)
