#include <iostream>
#include <fstream>
#include <opencv2/opencv.hpp>
#include <pcl/common/common.h>
#include <pcl/io/pcd_io.h>
#include <pcl/visualization/pcl_visualizer.h>
#include <Eigen/Eigen>
#include <iostream>
#include <filesystem>
#include <tuple>
#include <cmath>

#include "DataHandler.h"
#include "Eigen.h"
#include "VirtualSensor.h"
#include "SimpleMesh.h"
#include "FaceModel.h"
#include "ProcrustesAligner.h"

#define SHOW_MESH_CORRESPONDENCES 0

#define USE_POINT_TO_PLANE	1
#define USE_LINEAR_ICP		0

#define RUN_SHAPE_ICP		1
#define RUN_SEQUENCE_ICP	1

#define VISUALIZE 1


int main(int argc, char** argv){
    FaceModel model("../data");

	model.write_off(); 
    Data* data = read_dataset();


    std::vector<Vector3f> targetPoints = data->key_vectors;
    double target_scale = abs(targetPoints[1](0)-targetPoints[16](0));
    
    std::vector<Vector3f> sourcePoints = model.key_vectors;
    double source_scale = abs(sourcePoints[1](0)-sourcePoints[16](0));

    double scale = target_scale / source_scale ;

    for(int ind= 0 ; ind<sourcePoints.size() ; ind ++) {
        sourcePoints[ind] = scale * sourcePoints[ind] ;
    }

    std::vector<Vector3f> sp_deneme ;
    sp_deneme.push_back(sourcePoints[0]);sp_deneme.push_back(sourcePoints[16]);sp_deneme.push_back(sourcePoints[27]);sp_deneme.push_back(sourcePoints[8]) ;

    std::vector<Vector3f> tp_deneme ;
    tp_deneme.push_back(targetPoints[0]);tp_deneme.push_back(targetPoints[16]);tp_deneme.push_back(targetPoints[27]);tp_deneme.push_back(targetPoints[8]); ;

    ProcrustesAligner aligner;
	Matrix4f estimatedPose = aligner.estimatePose(sp_deneme, tp_deneme);

    if(VISUALIZE){
        // 3D Visualization:
        pcl::visualization::PCLVisualizer viewer("PCL Viewer");
        // Draw output point cloud:
        viewer.addCoordinateSystem (0.1);
        pcl::visualization::PointCloudColorHandlerRGBField<pcl::PointXYZRGB> rgb(data->cloud);
        viewer.addPointCloud<pcl::PointXYZRGB> (data->cloud, rgb, "cloud");
        pcl::PointCloud<pcl::PointXYZRGB>::Ptr point_to_visualize(new pcl::PointCloud<pcl::PointXYZRGB>);

        for(int k = 0 ; k<data->keypoints.size() ; k++){
            // Draw selected 3D point in red:
            pcl::PointXYZRGB keypoint = data->keypoints[k];
            keypoint.r = 255;
            keypoint.g = 0;
            keypoint.b = 0;
            point_to_visualize->points.push_back(keypoint);
            pcl::visualization::PointCloudColorHandlerRGBField<pcl::PointXYZRGB> red(point_to_visualize);
            viewer.addPointCloud<pcl::PointXYZRGB> (point_to_visualize, red, "kp_"+to_string(k));
            viewer.setPointCloudRenderingProperties (pcl::visualization::PCL_VISUALIZER_POINT_SIZE, 5, "kp_"+ std::to_string(k));

        }

        /* THE RAW KEY POINTS OF THE MODEL
        for(int k = 0 ; k<sourcePoints.size() ; k++){
            //blue
            pcl::PointXYZRGB keypoint =  pcl::PointXYZRGB(sourcePoints[k](0),sourcePoints[k](1),sourcePoints[k](2));
            keypoint.r = 0;
            keypoint.g = 0;
            keypoint.b = 255;
            point_to_visualize->points.push_back(keypoint);
            pcl::visualization::PointCloudColorHandlerRGBField<pcl::PointXYZRGB> green(point_to_visualize);
            viewer.addPointCloud<pcl::PointXYZRGB> (point_to_visualize, green, "model_kp_"+to_string(k));
            viewer.setPointCloudRenderingProperties (pcl::visualization::PCL_VISUALIZER_POINT_SIZE, 5, "model_kp_"+ std::to_string(k));

        }*/

        for(int k = 0 ; k<sourcePoints.size() ; k++){
            // green:
            Vector4f estimated(sourcePoints[k](0),sourcePoints[k](1),sourcePoints[k](2),1.0) ;
            estimated = estimatedPose * estimated ;
            pcl::PointXYZRGB keypoint =  pcl::PointXYZRGB(estimated(0),estimated(1),estimated(2));
            keypoint.r = 0;
            keypoint.g = 255;
            keypoint.b = 0;
            point_to_visualize->points.push_back(keypoint);
            pcl::visualization::PointCloudColorHandlerRGBField<pcl::PointXYZRGB> green(point_to_visualize);
            viewer.addPointCloud<pcl::PointXYZRGB> (point_to_visualize, green, "est_kp_"+to_string(k));
            viewer.setPointCloudRenderingProperties (pcl::visualization::PCL_VISUALIZER_POINT_SIZE, 5, "est_kp_"+ std::to_string(k));

        }


        viewer.setCameraPosition(-0.24917,-0.0187087,-1.29032, 0.0228136,-0.996651,0.0785278);
        // Loop for visualization (so that the visualizers are continuously updated):
        std::cout << "Visualization... "<< std::endl;
        while (not viewer.wasStopped())
        {
        viewer.spin();
        cv::waitKey(1);
        }
    }

}