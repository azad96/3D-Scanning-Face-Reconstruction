

#include <iostream>
#include <fstream>
#include <math.h>   


#include "VirtualSensor.h"
#include "SimpleMesh.h"
#include "ICPOptimizer.h"
#include "BridgeSparseDense.h"
#include <pcl/io/pcd_io.h>
#include <pcl/point_types.h>

#include <pcl/io/ply_io.h>
#include <pcl/io/obj_io.h>
#include <pcl/console/print.h>
#include <pcl/console/parse.h>
#include <pcl/console/time.h>
#include <pcl/surface/poisson.h>
#include <pcl/surface/grid_projection.h>
#include <pcl/io/vtk_lib_io.h>

using namespace pcl::io;
using namespace pcl::console;
using namespace std;

static const string METHOD_OF_POISSON = "poisson";
static const string METHOD_OF_GRID_PROJECTION = "grid_projection";

//default parameters
bool use_kd_tree = true;
//default poisson parameters
int depth = 8;
int solver_divide = 8;
int iso_divide = 8;
float point_weight = 4.0f;
//default grid projection parameters
float resolution = 0.005;
int padding_size = 3;
int nearest_neighbor_no = 10;
int max_binary_search_level = 10;



static void computePoisson(const pcl::PointCloud<pcl::PointNormal>::ConstPtr &input, pcl::PolygonMesh &output,
                           int depth, int solver_divide, int iso_divide, float point_weight, bool use_kd_tree) {
    print_info("Using parameters: depth %d, solver_divide %d, iso_divide %d, point_weight %d, use_kd_tree %d\n",
               depth, solver_divide, iso_divide, point_weight, use_kd_tree);
    pcl::Poisson<pcl::PointNormal> poisson;
    // Set parameters
    if(use_kd_tree) {
        // Create search tree*
        pcl::search::KdTree<pcl::PointNormal>::Ptr tree2(new pcl::search::KdTree<pcl::PointNormal>);
        tree2->setInputCloud(input);
        poisson.setSearchMethod(tree2);
    }
    poisson.setDepth(depth);
    poisson.setSolverDivide(solver_divide);
    poisson.setIsoDivide(iso_divide);
    poisson.setPointWeight(point_weight);
    poisson.setInputCloud(input);

    TicToc tt;
    print_highlight("Computing ...");
    tt.tic();
    poisson.reconstruct(output);
    print_info("[Done, ");
    print_value("%g", tt.toc());
    print_info(" ms]\n");
}

static void computeGridProjection(const pcl::PointCloud<pcl::PointNormal>::ConstPtr &input, pcl::PolygonMesh &output, float resolution,
                           int paddingSize, int nearestNeighborNo, int maxBinarySearchLevel, bool use_kd_tree) {

    print_info("Using parameters: resolution %.6f, padding size %d, nearest neighbor number %d, max binary search level %d, use_kd_tree %d\n",
               resolution, paddingSize, nearestNeighborNo, maxBinarySearchLevel, use_kd_tree);
    // Set parameters
    pcl::GridProjection<pcl::PointNormal> grid;
    if(use_kd_tree) {
        // Create search tree*
        pcl::search::KdTree<pcl::PointNormal>::Ptr tree2(new pcl::search::KdTree<pcl::PointNormal>);
        tree2->setInputCloud(input);
        grid.setSearchMethod(tree2);
    }
    grid.setResolution(resolution);
    grid.setPaddingSize(paddingSize);
    grid.setNearestNeighborNum(nearestNeighborNo);
    grid.setMaxBinarySearchLevel(maxBinarySearchLevel);
    grid.setInputCloud(input);

    TicToc tt;
    tt.tic();
    print_highlight("Computing ...");
    grid.reconstruct(output);
    print_info("[Done, ");
    print_value("%g", tt.toc());
    print_info(" ms]\n");

}

static void saveSurface(const string &filename, const pcl::PolygonMesh &input, bool isObj) {
    TicToc tt;
    tt.tic();

    print_highlight("Saving ");
    print_value("%s ", filename.c_str());

    if (isObj)
        saveOBJFile(filename, input);
    else
        savePolygonFileSTL(filename, input);

    print_info("[done, ");
    print_value("%g", tt.toc());
    print_info(" ms]\n");
}

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

    Data* data = read_dataset();

    Eigen::Matrix3f depthIntrinsics;
    depthIntrinsics << 1052.667867276341, 0, 962.4130834944134, 0, 1052.020917785721, 536.2206151001486, 0, 0, 1;

    Eigen::Matrix4f depthExtrinsics;
    depthExtrinsics.setIdentity();

    std::string pcd_filename="../dataset/cloud_1.pcd";

    // Load point cloud:
    pcl::PointCloud<pcl::PointXYZRGB>::Ptr cloud (new pcl::PointCloud<pcl::PointXYZRGB>);
    if (pcl::io::loadPCDFile<pcl::PointXYZRGB> (pcd_filename, *cloud) == -1) //* load the file
    {
        PCL_ERROR ("Couldn't read the pcd file \n");

    }

    std::vector<float> points;
    for (const auto& point : *cloud) {
        points.push_back(point.z);
    }
    std::cout << points.size() << std::endl;
    PointCloud target{ &points[0], depthIntrinsics, depthExtrinsics, 960, 540 };

    SimpleMesh targetMeshDeneme;
    Vertex v;
    for (const auto& point : target.getPoints()) {
        v.position << point, 1.0;
        v.color = Vector4uc(0, 0, 0, 0);
        targetMeshDeneme.addVertex(v);
    }

    pcl::PointCloud<pcl::PointNormal>::Ptr pcloud(new pcl::PointCloud<pcl::PointNormal>);
    // Fill in the cloud data
    pcloud->width    = 960;
    pcloud->height   = 540;
    pcloud->is_dense = true;
    pcloud->points.resize (pcloud->width * pcloud->height);
    unsigned i = 0;
    unsigned j = 0;
    for (auto& point: *pcloud)
    {
        if(target.getPoints()[i](2) < 0.1) {
            point.x = target.getPoints()[i](0);
            point.y = target.getPoints()[i](1);
            point.z = target.getPoints()[i](2);
            point.normal_x = target.getNormals()[i](0);
            point.normal_y = target.getNormals()[i](1);
            point.normal_z = target.getNormals()[i](2);
            j++;
        }
        i++;
    }
    std::cout << j << std::endl;
    pcl::PolygonMesh output;

    computePoisson(pcloud, output, depth, solver_divide, iso_divide, point_weight, use_kd_tree);
    //computeGridProjection(pcloud, output, resolution, padding_size, nearest_neighbor_no, max_binary_search_level, use_kd_tree);
    bool isObj = true;
    saveSurface("gp.obj", output, isObj);

    pcl::io::savePCDFileASCII ("test_pcd.pcd", *pcloud);




    // Estimate the pose from source to target mesh with ICP optimization.
    CeresICPOptimizer * optimizer = nullptr;
    optimizer = new CeresICPOptimizer();
    //optimizer->setMatchingMaxDistance(0.0003f);
    optimizer->setMatchingMaxDistance(0.3f);
    optimizer->setNbOfIterations(10);



    optimizer->estimateExpShapeCoeffs(target);

	delete optimizer;

	return 0;
}

int main() {
    std::cout << "Hello World!" << std::endl;
    std::string source_filename = "procrusted_face.off";

    write_prcrusted_mesh(source_filename);

    alignMeshWithICP();

    return 0;
}

