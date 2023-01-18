cmake_minimum_required(VERSION 2.8)
project(example)
find_package(PCL REQUIRED)
find_package(OpenCV REQUIRED)

add_definitions(${PCL_DEFINITIONS})
include_directories(${PCL_INCLUDE_DIRS} ${OpenCV_INCLUDE_DIRS})
link_directories(${PCL_LIBRARY_DIRS} ${OpenCV_LIB_DIR})

SET(EXECUTABLE_OUTPUT_PATH ${PROJECT_SOURCE_DIR}/bin)

add_executable (project3dtopixel src/project3dtopixel.cpp)
target_link_libraries(project3dtopixel ${PCL_LIBRARIES} ${OpenCV_LIBS})



