//
// Created by umur on 09.02.23.
//


#ifndef BUILD_BRIDGESPARSEDENSE_H
#define BUILD_BRIDGESPARSEDENSE_H

#endif //BUILD_BRIDGESPARSEDENSE_H
#include "PointCloud.h"
#include "FaceModel.h"
#include "Eigen.h"
#include "DataHandler.h"
#include "ProcrustesAligner.h"

#define VISUALIZE  1

static void write_prcrusted_mesh(std::string source_filename) {
    FaceModel* model = FaceModel::getInstance();

    Data* data = read_dataset();

    ProcrustesAligner aligner;

    //target from data
    std::vector<Vector3f> targetPoints = data->key_vectors;
    double target_scale = abs(targetPoints[1](0)-targetPoints[16](0));

    //source from face
    std::vector<Vector3f> sourcePoints = model->key_vectors;
    double source_scale = abs(sourcePoints[1](0)-sourcePoints[16](0));

    double scale = target_scale / source_scale ;

    Matrix4f scale_mat = Matrix4f::Identity() ;
    scale_mat(0,0) = scale ; scale_mat(1,1) = scale ; scale_mat(2,3) = scale ;

    for(int ind= 0 ; ind<sourcePoints.size() ; ind ++) {
        sourcePoints[ind] = scale * sourcePoints[ind] ;
    }

    model->scale_factor = scale;

    std::vector<Vector3f> sp_deneme ;
    sp_deneme.push_back(sourcePoints[0]);sp_deneme.push_back(sourcePoints[16]);sp_deneme.push_back(sourcePoints[27]);sp_deneme.push_back(sourcePoints[8]) ;

    std::vector<Vector3f> tp_deneme ;
    tp_deneme.push_back(targetPoints[0]);tp_deneme.push_back(targetPoints[16]);tp_deneme.push_back(targetPoints[27]);tp_deneme.push_back(targetPoints[8]); ;

    // estimated pose from source(face) to target (data)
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

        // THE RAW KEY POINTS OF THE MODEL
        /*for(int k = 0 ; k<sourcePoints.size() ; k++){
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
            pcl::PointXYZRGB transformed_face =  pcl::PointXYZRGB(estimated(0),estimated(1),estimated(2));

            std::cout << transformed_face.z << std::endl;

            transformed_face.r = 0;
            transformed_face.g = 255;
            transformed_face.b = 0;
            point_to_visualize->points.push_back(transformed_face);
            pcl::visualization::PointCloudColorHandlerRGBField<pcl::PointXYZRGB> green(point_to_visualize);
            viewer.addPointCloud<pcl::PointXYZRGB> (point_to_visualize, green, "est_kp_"+to_string(k));
            viewer.setPointCloudRenderingProperties (pcl::visualization::PCL_VISUALIZER_POINT_SIZE, 5, "est_kp_"+ std::to_string(k));

        }
        //std::cout << estimatedPose << std::endl;


        Matrix4d estimatedPoseD = estimatedPose.cast<double>();

        //std::cout << estimatedPoseD << std::endl;


        model->rotation = estimatedPoseD.block<3,3>(0,0);
        model->translation = estimatedPoseD.block<1,3>(3,0);
        Eigen::MatrixXd procrusted_mesh = model->transform(estimatedPoseD);
        model->write_off("../sample_face/" + source_filename, procrusted_mesh);

        std::cout << FaceModel::getInstance()->rotation << std::endl;

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