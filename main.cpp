#include <iostream>
#include <fstream>

#include "Eigen.h"
#include "VirtualSensor.h"
#include "SimpleMesh.h"
#include "ICPOptimizer.h"
#include "PointCloud.h"
#include "FaceModel.h"


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
    optimizer->setNbOfIterations(30);

	PointCloud target{ targetMesh };

	Matrix4d estimatedPose = Matrix4d::Identity();
    optimizer->estimateExpShapeCoeffs(target, estimatedPose);

	delete optimizer;

	return 0;
}

void createTransformedMesh()
{
	double r[6] = {0, 3.14/36, 0, 0, 0.2, 0};
	auto pose = PoseIncrement<double>(r);
	auto matrix = PoseIncrement<double>::convertToMatrix(pose);
	
	FaceModel* faceModel = FaceModel::getInstance();

	VectorXd random = VectorXd::Random(64)*2;
	faceModel->expCoefAr = random.data();
	faceModel->shapeCoefAr = random.data();

	auto mesh = faceModel->transform(matrix);
	faceModel->write_off("../sample_face/random.off", mesh);
}

int main() {

	createTransformedMesh();
	alignMeshWithICP();

    return 0;
}