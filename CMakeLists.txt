cmake_minimum_required(VERSION 3.6)

set(CMAKE_CONFIGURATION_TYPES Debug Release CACHE TYPE INTERNAL FORCE)

project(exercise_5)

# Options
set(LIBRARY_DIR ${CMAKE_CURRENT_SOURCE_DIR}/../Libs CACHE PATH "Path to lib folder")
set(Eigen3_DIR ${LIBRARY_DIR}/Eigen/share/eigen3/cmake CACHE PATH "Path to installed Eigen")
set(glog_DIR ${LIBRARY_DIR}/glog-lib/lib/cmake/glog CACHE PATH "Path to installed glog")
set(Ceres_DIR ${LIBRARY_DIR}/Ceres/CMake CACHE PATH "Path to installed Ceres")
set(Flann_INCLUDE_DIR ${LIBRARY_DIR}/Flann-1.8.4/ CACHE PATH "Path to Flann source folder")
set(HEADER_DIR ${CMAKE_CURRENT_SOURCE_DIR}/include CACHE PATH "Path to include folder")

# Use pre-compiled libraries on Windows
if(WIN32)
    if(CMAKE_SIZEOF_VOID_P EQUAL 8)
        set(FreeImage_INCLUDE_DIR ${LIBRARY_DIR}/FreeImage-3.18.0/Dist/x64/ CACHE PATH "Path to FreeImage header file")
        set(FreeImage_LIBRARY_DIR ${LIBRARY_DIR}/FreeImage-3.18.0/Dist/x64/ CACHE PATH "Path to FreeImage .lib/.dll folder")
    elseif(CMAKE_SIZEOF_VOID_P EQUAL 4)
        set(FreeImage_INCLUDE_DIR ${LIBRARY_DIR}/FreeImage-3.18.0/Dist/x32/ CACHE PATH "Path to FreeImage header file")
        set(FreeImage_LIBRARY_DIR ${LIBRARY_DIR}/FreeImage-3.18.0/Dist/x32/ CACHE PATH "Path to FreeImage .lib/.dll folder")
    endif()
endif(WIN32)

# Set C++ flags
set(CMAKE_CXX_STANDARD 14)
#SET(CMAKE_CXX_FLAGS ${CMAKE_CXX_FLAGS} " -g")

#find_package(nanoflann REQUIRED)
find_package(Ceres REQUIRED)
find_package(glog REQUIRED)
get_target_property(GLOG_DLL_PATH_DEBUG glog::glog IMPORTED_LOCATION_DEBUG)
get_target_property(GLOG_DLL_PATH_RELEASE glog::glog IMPORTED_LOCATION_RELEASE)

add_definitions("-D_DISABLE_EXTENDED_ALIGNED_STORAGE")

set(HEADER_FILES 
    ${HEADER_DIR}/Eigen.h 
    ${HEADER_DIR}/SimpleMesh.h 
    ${HEADER_DIR}/PointCloud.h 
    ${HEADER_DIR}/VirtualSensor.h 
    ${HEADER_DIR}/NearestNeighbor.h 
    ${HEADER_DIR}/ProcrustesAligner.h 
    ${HEADER_DIR}/ICPOptimizer.h 
    ${HEADER_DIR}/FreeImageHelper.h
)

set(SOURCE_FILES 
    src/FreeImageHelper.cpp
)

link_directories(${FreeImage_LIBRARY_DIR})
add_executable(out main.cpp ${HEADER_FILES} ${SOURCE_FILES})
target_link_libraries(out ceres freeimage Eigen3::Eigen)
target_include_directories(out PUBLIC ${HEADER_DIR} ${EIGEN3_INCLUDE_DIR} ${FreeImage_INCLUDE_DIR} ${Flann_INCLUDE_DIR} )
