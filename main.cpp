#include <iostream>
#include <fstream>
#include <opencv2/opencv.hpp>
#include <pcl/common/common.h>
#include <pcl/io/pcd_io.h>
#include <pcl/visualization/pcl_visualizer.h>
#include <Eigen/Eigen>
#include <dlib/image_processing/frontal_face_detector.h> 
#include <dlib/image_processing/render_face_detections.h>
#include <dlib/image_processing.h>
#include <dlib/gui_widgets.h>
#include <dlib/image_io.h>
#include <iostream>
#include <filesystem>
#include <tuple>
#include "project3dtopixel.h"


int main(int argc, char** argv){
    int x = read_dataset();
}