#include <opencv2/opencv.hpp>
#include <pcl/common/common.h>
#include <pcl/io/pcd_io.h>
#include <pcl/visualization/pcl_visualizer.h>
#include <Eigen/Eigen>

int main(int argc, char** argv)
{
    std::string filename = "../dataset/000_00_image.png";
    std::string pcd_filename=filename.substr(0,filename.find_last_of('_'))+"_cloud.pcd";
	// Load point cloud:
	pcl::PointCloud<pcl::PointXYZRGB>::Ptr cloud (new pcl::PointCloud<pcl::PointXYZRGB>);
	if (pcl::io::loadPCDFile<pcl::PointXYZRGB> (pcd_filename, *cloud) == -1) //* load the file
	{
		PCL_ERROR ("Couldn't read the pcd file \n");
		return (-1);
	}

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
    cv::imshow("prova",img);

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
