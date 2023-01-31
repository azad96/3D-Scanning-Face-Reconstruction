cmake_minimum_required(VERSION 3.6)
set(CMAKE_CONFIGURATION_TYPES Debug Release CACHE TYPE INTERNAL FORCE)

project(face_reconstruction)

# Set C++ flags
set(CMAKE_CXX_STANDARD 14)


if (${CMAKE_SYSTEM_NAME} MATCHES "Darwin")
    set(MACOSX TRUE)
endif()


if(MACOSX)
    set(Qt5_DIR /opt/homebrew/Cellar/qt@5/5.15.8_1/lib/cmake/Qt5)
    set(FreeImage_LIBRARY_DIR /opt/homebrew/Cellar/freeimage/3.18.0/lib) # set the lib path
    set(FreeImage_INCLUDE_DIR /opt/homebrew/Cellar/freeimage/3.18.0/include) # set the include path
endif(MACOSX) 

set(LIBRARY_DIR ${CMAKE_CURRENT_SOURCE_DIR}/../Libs CACHE PATH "Path to lib folder")
set(Eigen3_DIR ${LIBRARY_DIR}/Eigen/share/eigen3/cmake CACHE PATH "Path to installed Eigen")
set(Flann_INCLUDE_DIR ${LIBRARY_DIR}/Flann-1.8.4 CACHE PATH "Path to Flann source folder")


find_package(PCL)
find_package(PCL REQUIRED)
find_package(OpenCV REQUIRED)
find_package(dlib REQUIRED)
find_package(Eigen3 REQUIRED)



include_directories(
    ${PROJECT_SOURCE_DIR}/include
    ${PROJECT_SOURCE_DIR}/src
)

link_directories(${FreeImage_LIBRARY_DIR})


add_definitions(${PCL_DEFINITIONS})

include_directories(${PCL_INCLUDE_DIRS} ${OpenCV_INCLUDE_DIRS} )
link_directories(${PCL_LIBRARY_DIRS} ${OpenCV_LIB_DIR} ${dlib_LIB_DIR})

add_executable(${PROJECT_NAME} main.cpp ${HEADER_FILES} ${SOURCE_FILES})
target_include_directories(${PROJECT_NAME} PUBLIC 
    ${EIGEN3_INCLUDE_DIR} 
    ${FreeImage_INCLUDE_DIR} 
    ${Flann_INCLUDE_DIR}
)
target_link_libraries(${PROJECT_NAME}  freeimage ceres ${PCL_LIBRARIES} ${OpenCV_LIBS} dlib::dlib Eigen3::Eigen)


