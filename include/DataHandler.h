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
#include <pcl/filters/crop_box.h>
#include "Eigen.h"
#include "PointCloud.h"


#define WRITE_TXT 0

using namespace std;

class Data {      
  public:            
    std::vector<pcl::PointXYZRGB> keypoints;        
    pcl::PointCloud<pcl::PointXYZRGB>::Ptr cloud;  
    std::vector<Vector3f>  key_vectors;
    cv::Mat img ;
    PointCloud cropped_cloud;
    PointCloud fullCloud;
    std::vector<Eigen::Vector3f> pclPoints;

    Data(std::vector<pcl::PointXYZRGB> keypoints,pcl::PointCloud<pcl::PointXYZRGB>::Ptr cloud,std::vector<Eigen::Vector3f> vec,cv::Mat img,PointCloud cropped_cloud, PointCloud fullCloud, std::vector<Eigen::Vector3f> pclPoints) {   
      this->keypoints = keypoints ;
      this->cloud = cloud;
      this->key_vectors = vec ;
      this->img = img;
      this->cropped_cloud = cropped_cloud;
      this->fullCloud = fullCloud;
      this->pclPoints = pclPoints;
    }
};


Data* read_dataset()
{
    //std::string filename = "../dataset/image_1.png";
    //std::string pcd_filename="../dataset/cloud_1.pcd";
    
    std::string filename = "../RGBD_Face_dataset_testing/Test3/001_06_image.png";
    std::string pcd_filename="../RGBD_Face_dataset_testing/Test3/001_06_cloud.pcd";

    //std::string filename = "../RGBD_Face_dataset_training/000_05_image.png";
    //std::string pcd_filename="../RGBD_Face_dataset_training/000_05_cloud.pcd";

	// Load point cloud:
	pcl::PointCloud<pcl::PointXYZRGB>::Ptr cloud (new pcl::PointCloud<pcl::PointXYZRGB>);
	if (pcl::io::loadPCDFile<pcl::PointXYZRGB> (pcd_filename, *cloud) == -1) //* load the file
	{
		PCL_ERROR ("Couldn't read the pcd file \n");
		return nullptr;
	}

    // We need a face detector.  We will use this to get bounding boxes for
    // each face in an image.  
    dlib::frontal_face_detector detector = dlib::get_frontal_face_detector();

    // And we also need a shape_predictor.  This is the tool that will predict face
    // landmark positions given an image and face bounding box.  Here we are just
    // loading the model from the shape_predictor_68_face_landmarks.dat file you gave
    // as a command line argument.
    dlib::shape_predictor sp;
    dlib::deserialize("../models/shape_predictor_68_face_landmarks.dat") >> sp;
    //dlib::image_window win, win_faces;
    cout << "processing image " << filename << endl;

    dlib::array2d<dlib::rgb_pixel> img_kp;
    dlib::load_image(img_kp, filename);

    // Now tell the face detector to give us a list of bounding boxes
    // around all the faces in the image.
    std::vector<dlib::rectangle> dets = detector(img_kp);

    // Now we will go ask the shape_predictor to tell us the pose of
    // each face we detected.
    std::vector<dlib::full_object_detection> shapes;
    for (unsigned long j = 0; j < dets.size(); ++j)
    {
        dlib::full_object_detection shape = sp(img_kp, dets[j]);
        
        std::string s = string(filename);

        size_t i = s.rfind('.', s.length());
        std::string im_path = s.substr(0,i);

        std::ofstream outfile (im_path+".txt");
        for ( int l = 0 ; l <shape.num_parts() ; l++){
            outfile << shape.part(l) << std::endl;
        }
        outfile.close();
        // You get the idea, you can get all the face part locations if
        // you want them.  Here we just store them in shapes so we can
        // put them on the screen.
        shapes.push_back(shape);
        //TODO: only one face will be processed
        break;
    }

    // Now let's view our face poses on the screen.
    /*win.clear_overlay();
    win.set_image(img_kp);
    win.add_overlay(render_face_detections(shapes));*/

    // We can also extract copies of each face that are cropped, rotated upright,
    // and scaled to a standard size as shown here:
    dlib::array<dlib::array2d<dlib::rgb_pixel> > face_chips;
    dlib::extract_image_chips(img_kp, dlib::get_face_chip_details(shapes), face_chips);
    //cv::Mat img_clip = dlib::toMat(std::move(face_chips[0]));

    // Load corresponding image:
    cv::Mat img = cv::imread(filename, cv::IMREAD_COLOR);

    // Define projection matrix from 3D to 2D:
    //P matrix is in camera_info.yaml
    Eigen::Matrix<float, 3,4> P;
    P <<  1052.667867276341, 0, 962.4130834944134, 0, 0, 1052.020917785721, 536.2206151001486, 0, 0, 0, 1, 0;

    // Select a 3D point in the point cloud with row and column indices:

    std::vector<pcl::PointXYZRGB> keypoints ;
    vector<int> x_val;
    vector<int> y_val;

    std::vector<Vector3f> keypoints_vectors ; 
    for ( int l = 0 ; l <shapes[0].num_parts() ; l++){
        pcl::PointXYZRGB keypoint = cloud->at((int)shapes[0].part(l).x()/2, (int)shapes[0].part(l).y()/2);

        x_val.push_back(shapes[0].part(l).x());
        y_val.push_back(shapes[0].part(l).y());
        keypoints.push_back(keypoint);
        Vector3f a(keypoint.x, keypoint.y, keypoint.z);
        keypoints_vectors.push_back(a);
    }
    int min_x = dets[0].tl_corner().x() ;
    int min_y = dets[0].tl_corner().y() ;
    int max_x = dets[0].br_corner().x() ;
    int max_y = dets[0].br_corner().y() ;

    //float* cropped_depth_map = new float[(max_x-min_x)*(max_y-min_y)];
    float* cropped_depth_map = new float[540*960];
    int ind =0 ;
    for(int y = 0; y< 540 ;y++){
        for(int x =0 ; x<960 ; x++){
            cropped_depth_map[ind] = cloud->at(x,y).z;
            ind++;
        }

    }
    
    Eigen::Matrix3f depthIntrinsics;
    Eigen::Matrix4f depthExtrinsics;
    depthExtrinsics.setIdentity();
    depthIntrinsics <<  1052.667867276341/2, 0, 962.4130834944134/2, 0, 1052.020917785721/2, 536.2206151001486/2, 0, 0,1;

    PointCloud fullCloud;

    std::vector<Eigen::Vector3f> pclPoints;
    std::vector<Eigen::Vector3f> rgb_;

    int null_count = 0;
    int total_count = 0;
    // for (const auto& point : *cloud) {
    for ( int y = 0 ; y <540 ; y++){
        for ( int x = 0 ; x <960 ; x++){
            pcl::PointXYZRGB keypoint = cloud->at(x, y);
            
            Vector3f a(keypoint.x, keypoint.y, keypoint.z);
            pclPoints.push_back(a);
            Vector3f b(keypoint.r, keypoint.g, keypoint.b);
            rgb_.push_back(b);

        }
    }

    PointCloud cropped_cloud = PointCloud(cropped_depth_map,pclPoints,960,540,rgb_);

    return new Data(keypoints,cloud,keypoints_vectors,img,cropped_cloud,fullCloud, pclPoints);

}
