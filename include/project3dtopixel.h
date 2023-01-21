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

using namespace dlib;
using namespace std;

int read_dataset()
{
    //std::string filename = "../dataset/000_00_image.png";
    //std::string pcd_filename=filename.substr(0,filename.find_last_of('_'))+"_cloud.pcd";
    std::string filename = "../dataset/image_1.png";
    std::string pcd_filename="../dataset/cloud_1.pcd";

	// Load point cloud:
	pcl::PointCloud<pcl::PointXYZRGB>::Ptr cloud (new pcl::PointCloud<pcl::PointXYZRGB>);
	if (pcl::io::loadPCDFile<pcl::PointXYZRGB> (pcd_filename, *cloud) == -1) //* load the file
	{
		PCL_ERROR ("Couldn't read the pcd file \n");
		return (-1);
	}
    // We need a face detector.  We will use this to get bounding boxes for
    // each face in an image.  
    frontal_face_detector detector = get_frontal_face_detector();
    // And we also need a shape_predictor.  This is the tool that will predict face
    // landmark positions given an image and face bounding box.  Here we are just
    // loading the model from the shape_predictor_68_face_landmarks.dat file you gave
    // as a command line argument.
    shape_predictor sp;
    deserialize("../models/shape_predictor_68_face_landmarks.dat") >> sp;
    image_window win, win_faces;
    cout << "processing image " << filename << endl;

    array2d<rgb_pixel> img_kp;
    load_image(img_kp, filename);
    // TODO : pyramid up i yoruma aldim ama belki daha sonra pyramid down deriz
    // Make the image larger so we can detect small faces.
    //pyramid_up(img_kp);

    // Now tell the face detector to give us a list of bounding boxes
    // around all the faces in the image.
    std::vector<rectangle> dets = detector(img_kp);
    cout << "Number of faces detected: " << dets.size() << endl;

    // Now we will go ask the shape_predictor to tell us the pose of
    // each face we detected.
    std::vector<full_object_detection> shapes;
    for (unsigned long j = 0; j < dets.size(); ++j)
    {
        full_object_detection shape = sp(img_kp, dets[j]);
        cout << "number of parts: "<< shape.num_parts() << endl;
        cout << "pixel position of first part:  " << shape.part(0) << endl;
        cout << "pixel position of second part: " << shape.part(1) << endl;
        
        std::string s = string(filename);

        size_t i = s.rfind('.', s.length());
        std::string im_path = s.substr(0,i);

        std::cout<<im_path<<std::endl;
        std::ofstream outfile (im_path+".txt");
        for ( int l = 0 ; l <shape.num_parts() ; l++){
            //cout << "landmark " <<  l << " " << shape.part(l) << endl;
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
    win.clear_overlay();
    win.set_image(img_kp);
    win.add_overlay(render_face_detections(shapes));

    // We can also extract copies of each face that are cropped, rotated upright,
    // and scaled to a standard size as shown here:
    dlib::array<array2d<rgb_pixel> > face_chips;
    extract_image_chips(img_kp, get_face_chip_details(shapes), face_chips);
    win_faces.set_image(tile_images(face_chips));

    //cout << "Hit enter to process the next image..." << endl;
    //TODO delete below line
    //cin.get();


    // Load corresponding image:
    cv::Mat img = cv::imread(filename, cv::IMREAD_COLOR);

    // Select a 3D point in the point cloud with row and column indices:
    pcl::PointXYZRGB selected_point = cloud->at(380, 275);
    std::cout << "Selected 3D point: " << selected_point.x << " " << selected_point.y << " " << selected_point.z << std::endl;

    // Define projection matrix from 3D to 2D:
    //P matrix is in camera_info.yaml
    Eigen::Matrix<float, 3,4> P;
    P <<  1052.667867276341, 0, 962.4130834944134, 0, 0, 1052.020917785721, 536.2206151001486, 0, 0, 0, 1, 0;

    // 3D to 2D projection:
    //Let's do P*point and rescale X,Y
    Eigen::Vector4f homogeneous_point(selected_point.x, selected_point.y, selected_point.z,1);
    Eigen::Vector3f output = P * homogeneous_point;
    output[0] /= output[2];
    output[1] /= output[2];

    std::cout << "Corresponding 2D point in the image: " << output(0) << " " << output(1) << std::endl;

    // Draw a circle around the 2d point:
    cv::circle(img,cv::Point(output[0],output[1]),10,cv::Scalar(0,0,255), 3);
    //cv::imshow("prova",img);
    //cv::imwrite("prova.png",img);

    // 3D Visualization:
    pcl::visualization::PCLVisualizer viewer("PCL Viewer");

    // Draw output point cloud:
    viewer.addCoordinateSystem (0.1);
    pcl::visualization::PointCloudColorHandlerRGBField<pcl::PointXYZRGB> rgb(cloud);
    viewer.addPointCloud<pcl::PointXYZRGB> (cloud, rgb, "cloud");

    // Draw selected 3D point in red:
    selected_point.r = 255;
    selected_point.g = 0;
    selected_point.b = 0;
    pcl::PointCloud<pcl::PointXYZRGB>::Ptr point_to_visualize(new pcl::PointCloud<pcl::PointXYZRGB>);
    point_to_visualize->points.push_back(selected_point);
    pcl::visualization::PointCloudColorHandlerRGBField<pcl::PointXYZRGB> red(point_to_visualize);
    viewer.addPointCloud<pcl::PointXYZRGB> (point_to_visualize, red, "point");
    viewer.setPointCloudRenderingProperties (pcl::visualization::PCL_VISUALIZER_POINT_SIZE, 15, "point");
    viewer.setCameraPosition(-0.24917,-0.0187087,-1.29032, 0.0228136,-0.996651,0.0785278);

    // Loop for visualization (so that the visualizers are continuously updated):
    std::cout << "Visualization... "<< std::endl;
    while (not viewer.wasStopped())
    {
      viewer.spin();
      cv::waitKey(1);
    }

}
