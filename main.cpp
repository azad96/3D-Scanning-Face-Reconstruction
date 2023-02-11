#include <iostream>
#include <fstream>
#include <math.h>   
#include <pcl/io/vtk_lib_io.h>
#include <pcl/io/obj_io.h>

#include "Eigen.h"
#include "VirtualSensor.h"
#include "SimpleMesh.h"
#include "ICPOptimizer.h"
#include "PointCloud.h"
#include "FaceModel.h"
#include "DataHandler.h"

#define VISUALIZE  1
using namespace std;

int alignMeshWithICP() {
	// Load the source and target mesh.
	const std::string filenameSource = std::string("../sample_face/neutral.off");
	const std::string filenameTarget = std::string("../sample_face/random.off");

	SimpleMesh sourceMesh;
	if (!sourceMesh.loadMesh(filenameSource)) {
		std::cout << "Mesh file wasn't read successfully at location: " << filenameSource << std::endl;
		return -1;
	}

	SimpleMesh targetMesh;
	if (!targetMesh.loadMesh(filenameTarget)) {
		std::cout << "Mesh file wasn't read successfully at location: " << filenameTarget << std::endl;
		return -1;
	}

	// Estimate the pose from source to target mesh with ICP optimization.
    CeresICPOptimizer * optimizer = nullptr;
    optimizer = new CeresICPOptimizer();
    //optimizer->setMatchingMaxDistance(0.0003f);
    optimizer->setMatchingMaxDistance(0.000003f);
    optimizer->setNbOfIterations(10);
	PointCloud target{ targetMesh };
    optimizer->estimateExpShapeCoeffs(target);
	delete optimizer;

	return 0;
}

int alignMeshWithICP(PointCloud target) {
	// Load the source and target mesh.
	const std::string filenameSource = std::string("../sample_face/neutral.off");
	

	SimpleMesh sourceMesh;
	if (!sourceMesh.loadMesh(filenameSource)) {
		std::cout << "Mesh file wasn't read successfully at location: " << filenameSource << std::endl;
		return -1;
	}



	// Estimate the pose from source to target mesh with ICP optimization.
    CeresICPOptimizer * optimizer = nullptr;
    optimizer = new CeresICPOptimizer();
    //optimizer->setMatchingMaxDistance(0.0003f);
    optimizer->setMatchingMaxDistance(0.1f);
    optimizer->setNbOfIterations(10);
    optimizer->estimateExpShapeCoeffs(target);
	delete optimizer;

	return 0;
} 


int main() {

	FaceModel* model = FaceModel::getInstance();

	Data* data = read_dataset();

	ProcrustesAligner aligner;

	std::vector<Vector3f> targetPoints = data->key_vectors;
	double target_scale = abs(targetPoints[1](0)-targetPoints[16](0));

	std::vector<Vector3f> sourcePoints = model->key_vectors;
    double source_scale = abs(sourcePoints[1](0)-sourcePoints[16](0));
	
	double scale = target_scale / source_scale ;

    for(int ind= 0 ; ind<sourcePoints.size() ; ind ++) {
        sourcePoints[ind] = scale * sourcePoints[ind] ;
    }

    std::vector<Vector3f> sp_deneme ;
    sp_deneme.push_back(sourcePoints[0]);sp_deneme.push_back(sourcePoints[16]);sp_deneme.push_back(sourcePoints[27]);sp_deneme.push_back(sourcePoints[8]) ;

    std::vector<Vector3f> tp_deneme ;
    tp_deneme.push_back(targetPoints[0]);tp_deneme.push_back(targetPoints[16]);tp_deneme.push_back(targetPoints[27]);tp_deneme.push_back(targetPoints[8]); ;

    
	Matrix4f estimatedPose = aligner.estimatePose(sp_deneme, tp_deneme);

    

    Matrix4d estimatedPoseD = estimatedPose.cast<double>();

    model->pose = estimatedPoseD;
    model->scale = scale;
    model->rotation = estimatedPoseD.block<3,3>(0,0);
    model->translation = estimatedPoseD.block<3,1>(0,3);


    if(VISUALIZE){
        // 3D Visualization:
        pcl::visualization::PCLVisualizer viewer("PCL Viewer");
        // Draw output point cloud:     
        viewer.addCoordinateSystem (0.1);
        pcl::visualization::PointCloudColorHandlerRGBField<pcl::PointXYZRGB> rgb(data->cloud);
        viewer.addPointCloud<pcl::PointXYZRGB> (data->cloud, rgb, "cloud");
        pcl::PointCloud<pcl::PointXYZRGB>::Ptr point_to_visualize(new pcl::PointCloud<pcl::PointXYZRGB>);

    /*
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

        }*/

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
        /*
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

        }*/

        /*
        for(int k=0 ; data->cropped_cloud.getPoints().size() ; k++){
            auto p = data->cropped_cloud.getPoints()[k];
            pcl::PointXYZRGB cp = pcl::PointXYZRGB(p(0),p(1),p(2));
            cp.r=0,cp.g=0,cp.b=255;
            point_to_visualize->points.push_back(cp);
            //viever.addPointCloud<pcl::PointXYZRGB> (point_to_visualize,  )
            pcl::visualization::PointCloudColorHandlerRGBField<pcl::PointXYZRGB> blue(point_to_visualize);
            viewer.addPointCloud<pcl::PointXYZRGB> (point_to_visualize, blue, "cropped_"+to_string(k));
            viewer.setPointCloudRenderingProperties (pcl::visualization::PCL_VISUALIZER_POINT_SIZE, 5, "cropped_"+ std::to_string(k));

        }*/


        MatrixXd transformed_mesh;
        transformed_mesh = model->transform(model->pose, model->scale);
        model->write_obj("../sample_face/transformed_model.obj",transformed_mesh);
        model->write_off("../sample_face/transformed_model.off",transformed_mesh);

        pcl::PolygonMesh pcl_mesh;
        pcl::io::loadOBJFile("../sample_face/transformed_model.obj",pcl_mesh); 


        viewer.addPolygonMesh(pcl_mesh,"pcl_mesh",0);


        viewer.setCameraPosition(-0.24917,-0.0187087,-1.29032, 0.0228136,-0.996651,0.0785278);
        // Loop for visualization (so that the visualizers are continuously updated):
        std::cout << "Visualization... "<< std::endl;
        alignMeshWithICP(data->cropped_cloud);
        while (not viewer.wasStopped())
        {
        viewer.spin();
        cv::waitKey(1);
        }
    }

    
    

    return 0;
}