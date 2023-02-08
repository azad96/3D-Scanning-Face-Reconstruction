cmake_minimum_required(VERSION 3.6)

set(CMAKE_CONFIGURATION_TYPES Debug Release CACHE TYPE INTERNAL FORCE)

project(face_reconstruction)

# Options
set(LIBRARY_DIR ${CMAKE_CURRENT_SOURCE_DIR}/libs CACHE PATH "Path to lib folder")
set(Eigen3_DIR ${LIBRARY_DIR}/Eigen/share/eigen3/cmake CACHE PATH "Path to installed Eigen")
set(glog_DIR ${LIBRARY_DIR}/Glog/lib/cmake/glog CACHE PATH "Path to installed glog")
set(Ceres_DIR ${LIBRARY_DIR}/Ceres/lib/cmake/Ceres CACHE PATH "Path to installed Ceres")
set(Flann_INCLUDE_DIR ${LIBRARY_DIR}/Flann-1.8.4 CACHE PATH "Path to Flann source folder")
set(GLFW_INCLUDE_DIRS ${CMAKE_CURRENT_SOURCE_DIR}/glfw/include CACHE PATH "Path to glfw include")
set(GLFW_LIBRARY ${CMAKE_CURRENT_SOURCE_DIR}/glfw/lib/libglfw.3.dylib CACHE PATH "Path to glfw include")


if (${CMAKE_SYSTEM_NAME} MATCHES "Darwin")
    set(MACOSX TRUE)
endif()

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

if(MACOSX)
    set(Qt5_DIR /opt/homebrew/Cellar/qt@5/5.15.8_1/lib/cmake/Qt5)
    set(FreeImage_LIBRARY_DIR /opt/homebrew/Cellar/freeimage/3.18.0/lib) # set the lib path
    set(FreeImage_INCLUDE_DIR /opt/homebrew/Cellar/freeimage/3.18.0/include) # set the include path
endif(MACOSX)

# Set C++ flags
set(CMAKE_CXX_STANDARD 14)

find_package(PCL)
find_package(PCL REQUIRED)
find_package(Ceres REQUIRED)
find_package(glog REQUIRED)
find_package(OpenCV REQUIRED)
find_package(dlib REQUIRED)
find_package(glew REQUIRED)

get_target_property(GLOG_DLL_PATH_DEBUG glog::glog IMPORTED_LOCATION_DEBUG)
get_target_property(GLOG_DLL_PATH_RELEASE glog::glog IMPORTED_LOCATION_RELEASE)

add_definitions("-D_DISABLE_EXTENDED_ALIGNED_STORAGE")
add_definitions(${PCL_DEFINITIONS})

include_directories(
    ${PROJECT_SOURCE_DIR}/include
    ${PROJECT_SOURCE_DIR}/src
)
include_directories(${PCL_INCLUDE_DIRS} ${OpenCV_INCLUDE_DIRS} )
link_directories(${PCL_LIBRARY_DIRS} ${OpenCV_LIB_DIR} ${dlib_LIB_DIR})


# Set files to be compiled
file(GLOB HEADER_FILES 
    ${PROJECT_SOURCE_DIR}/include/*.h
)
file(GLOB SOURCE_FILES 
    ${PROJECT_SOURCE_DIR}/src/*.cpp
)

link_directories(${FreeImage_LIBRARY_DIR})
add_executable(${PROJECT_NAME} main.cpp ${HEADER_FILES} ${SOURCE_FILES})
target_link_libraries(${PROJECT_NAME} freeimage ceres ${PCL_LIBRARIES} ${OpenCV_LIBS} dlib::dlib Eigen3::Eigen ${GLFW_LIBRARY} ${GLEW_LIBRARY} )
# target_link_libraries(${PROJECT_NAME} freeimage glog::glog ceres Eigen3::Eigen)
target_include_directories(${PROJECT_NAME} PUBLIC 
    ${EIGEN3_INCLUDE_DIR} 
    ${FreeImage_INCLUDE_DIR} 
    ${Flann_INCLUDE_DIR}
    ${GLEW_INCLUDE_DIRS} 
    ${GLFW_INCLUDE_DIRS}
)


# On Windows copy dll to executable folder
if(WIN32)
    # Visual Studio properties
    set_property(DIRECTORY ${CMAKE_CURRENT_SOURCE_DIR} PROPERTY VS_STARTUP_PROJECT ${PROJECT_NAME})
    set_property(TARGET ${PROJECT_NAME} PROPERTY VS_DEBUGGER_WORKING_DIRECTORY ${CMAKE_BINARY_DIR}/)

    # On Windows copy dll to executable folder
    add_custom_command(TARGET ${PROJECT_NAME} POST_BUILD
        COMMAND ${CMAKE_COMMAND} -E copy_if_different
            "${FreeImage_LIBRARY_DIR}/FreeImage.dll"
            ${CMAKE_BINARY_DIR})
    
    add_custom_command(TARGET ${PROJECT_NAME} POST_BUILD        
        COMMAND ${CMAKE_COMMAND} -E copy_if_different          
            "${GLOG_DLL_PATH_DEBUG}"                           
            ${CMAKE_BINARY_DIR})                     
    
    add_custom_command(TARGET ${PROJECT_NAME} POST_BUILD
        COMMAND ${CMAKE_COMMAND} -E copy_if_different
            "${GLOG_DLL_PATH_RELEASE}"
            ${CMAKE_BINARY_DIR})
endif(WIN32)