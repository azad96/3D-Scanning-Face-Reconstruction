cmake_minimum_required(VERSION 3.6)
set(CMAKE_CONFIGURATION_TYPES Debug Release CACHE TYPE INTERNAL FORCE)

project(face_reconstruction)

# Set C++ flags
set(CMAKE_CXX_STANDARD 14)

find_package(PCL REQUIRED)
find_package(OpenCV REQUIRED)
find_package(dlib REQUIRED)


include_directories(
    ${PROJECT_SOURCE_DIR}/include
    ${PROJECT_SOURCE_DIR}/src
)

add_definitions(${PCL_DEFINITIONS})

include_directories(${PCL_INCLUDE_DIRS} ${OpenCV_INCLUDE_DIRS} )
link_directories(${PCL_LIBRARY_DIRS} ${OpenCV_LIB_DIR} ${dlib_LIB_DIR})

add_executable(${PROJECT_NAME} main.cpp ${HEADER_FILES} ${SOURCE_FILES})

target_link_libraries(${PROJECT_NAME}  ${PCL_LIBRARIES} ${OpenCV_LIBS} dlib::dlib)



