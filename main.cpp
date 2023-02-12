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
    optimizer->setMatchingMaxDistance(0.0000003f);
    optimizer->setNbOfIterations(15);
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
    optimizer->setMatchingMaxDistance(0.000005f);
    optimizer->setNbOfIterations(60);
    optimizer->estimateExpShapeCoeffs(target);
	delete optimizer;

	return 0;
} 

int alignMeshWithICP(std::vector<Eigen::Vector3f> target) {
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
    optimizer->setMatchingMaxDistance(0.003f);
    //optimizer->setMatchingMaxDistance(1.0f);
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
	std::vector<Vector3f> sourcePoints = model->key_vectors;

	
	Eigen::Vector3f target_diff;
    target_diff << targetPoints[1](0) - targetPoints[16](0),	targetPoints[1](1)-targetPoints[16](1), targetPoints[1](2)-targetPoints[16](2);
	Eigen::Vector3f source_diff;
    source_diff << sourcePoints[1](0) - sourcePoints[16](0), 	sourcePoints[1](1)-sourcePoints[16](1), sourcePoints[1](2)-sourcePoints[16](2);
	
	double target_scale = target_diff.norm();
    double source_scale = source_diff.norm();
	
	double scale = target_scale / source_scale;

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

    FaceModel* faceModel = FaceModel::getInstance();
    if(VISUALIZE){
        // 3D Visualization:
        pcl::visualization::PCLVisualizer viewer("PCL Viewer");
        // Draw output point cloud:     
        viewer.addCoordinateSystem (0.1);
        pcl::visualization::PointCloudColorHandlerRGBField<pcl::PointXYZRGB> rgb(data->cloud);
        viewer.addPointCloud<pcl::PointXYZRGB> (data->cloud, rgb, "cloud");
        pcl::PointCloud<pcl::PointXYZRGB>::Ptr point_to_visualize(new pcl::PointCloud<pcl::PointXYZRGB>);

    
        // for(int k = 0 ; k<data->keypoints.size() ; k++){
        //     // Draw selected 3D point in red:
        //     pcl::PointXYZRGB keypoint = data->keypoints[k];
        //     keypoint.r = 255;
        //     keypoint.g = 0;
        //     keypoint.b = 0;
        //     point_to_visualize->points.push_back(keypoint);
        //     pcl::visualization::PointCloudColorHandlerRGBField<pcl::PointXYZRGB> red(point_to_visualize);
        //     viewer.addPointCloud<pcl::PointXYZRGB> (point_to_visualize, red, "kp_"+to_string(k));
        //     viewer.setPointCloudRenderingProperties (pcl::visualization::PCL_VISUALIZER_POINT_SIZE, 5, "kp_"+ std::to_string(k));

        // }

        

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
        std::cout << "Basladi "<< std::endl;
        
        //vector<Vector3f> points_ = data->cropped_cloud.getPoints();
        //vector<Vector3f> normals_ = data->cropped_cloud.getNormals();
        //  for(int k=0 ; k < points_.size() ; k=k+50){
        //      auto p = points_[k];
        //      Vector3f rgb_value = data->cropped_cloud.rgb[k];
        //      Vector3f n_value = normals_[k];
        //      pcl::PointXYZRGB cp = pcl::PointXYZRGB(p(0),p(1),p(2),rgb_value(0), rgb_value(1), rgb_value(2));

        //      pcl::PointXYZRGB n = pcl::PointXYZRGB(n_value(0)+p(0),n_value(1)+p(1),n_value(2)+p(2),0, 255, 0);
        //      //cp.r=0,cp.g=0,cp.b=255;
        //      if(k/50%30==0)
        //         viewer.addLine(cp, n, 0, 255, 0,"line_"+to_string(k));
        //      /*cp.r = rgb_value(0); cp.g = rgb_value(1); cp.b = rgb_value(2);*/
        //      point_to_visualize->points.push_back(cp);
        //      //viever.addPointCloud<pcl::PointXYZRGB> (point_to_visualize,  )
        //      pcl::visualization::PointCloudColorHandlerRGBField<pcl::PointXYZRGB> blue(point_to_visualize);
        //      viewer.addPointCloud<pcl::PointXYZRGB> (point_to_visualize, blue, "cropped_"+to_string(k));
        //      viewer.setPointCloudRenderingProperties (pcl::visualization::PCL_VISUALIZER_POINT_SIZE, 5, "cropped_"+ std::to_string(k));
        //  }

        /*for(int k=0 ; k < data->fullCloud.getPoints().size() ; k++){
            auto p = data->fullCloud.getPoints()[k];
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

        


        
        // std::unique_ptr<NearestNeighborSearch> m_nearestNeighborSearch = std::make_unique<NearestNeighborSearchFlann>();
        
        // std::vector<Eigen::Vector3f> target = data->pclPoints;
        std::vector<Eigen::Vector3f> source = data->key_vectors;
        // m_nearestNeighborSearch->setMatchingMaxDistance(0.003f);
        // m_nearestNeighborSearch->buildIndex(target);

        // cout << "print 1 " << endl;
        // cout << target[960*238 + 338] << endl << "dsa" << endl;
        // source[0](0) += 0.00002;
        // cout << source[0] << endl << "asd" << endl;

        // auto matches = m_nearestNeighborSearch->queryMatches(source);
        // int i = 0;
        // for(Match match : matches) {
        //     cout << match.idx << endl;
        //     if (match.idx >= 0) {
        //         cout << "girdim" << endl;
        //         const auto &targetPoint = target[match.idx];
        //         const int sourcePointIndex = i;

        //         cout << "Point" << endl;
        //         cout << targetPoint << endl;
        //         cout << source[sourcePointIndex] << endl;
        //     }
        //     i++;
        // }
        //MatrixXd transformed_mesh;
        SimpleMesh faceMesh;
        if (!faceMesh.loadMesh("../sample_face/transformed_model.off")) {
            std::cout << "Mesh file wasn't read successfully at location: " << "transformed_model.off" << std::endl;
        }


        PointCloud faceModelPoints{faceMesh};
        // vector<Vector3f> points = faceModelPoints.getPoints();
        // cout << "hey" << faceModel->key_points[31] << endl;
        // transformed_mesh = model->transform(model->pose, model->scale);
        // cout << " key point " << (transformed_mesh.block(faceModel->key_points[31], 0, 1, 3)) << endl;
        // cout << " source point " << source[31] << endl;
        // cout << " read mesh " << points[faceModel->key_points[31]] << endl;
        // cout << "distance " << (source[31] - points[faceModel->key_points[31]]).norm() << endl;
        
        alignMeshWithICP(data->cropped_cloud);
        
        pcl::PolygonMesh pcl_mesh;
        pcl::io::loadOBJFile("../sample_face/result.obj",pcl_mesh); 

        viewer.addPolygonMesh(pcl_mesh,"pcl_mesh",0);

        viewer.setCameraPosition(-0.24917,-0.0187087,-1.29032, 0.0228136,-0.996651,0.0785278);
        // Loop for visualization (so that the visualizers are continuously updated):
        std::cout << "Visualization... "<< std::endl;
        
        while (not viewer.wasStopped())
        {
            viewer.spin();
            cv::waitKey(1);
        }
    }

    
    

    return 0;
}