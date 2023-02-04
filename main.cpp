#include <iostream>
#include <fstream>

#include "Eigen.h"
#include "VirtualSensor.h"
#include "SimpleMesh.h"
#include "ICPOptimizer.h"
#include "PointCloud.h"
#include "FaceModel.h"


void debugCorrespondenceMatching() {
	// Load the source and target mesh.
	const std::string filenameSource = std::string("source_mesh.off");
	const std::string filenameTarget = std::string("target_mesh.off");

	SimpleMesh sourceMesh;
	if (!sourceMesh.loadMesh(filenameSource)) {
		std::cout << "Mesh file wasn't read successfully." << std::endl;
		return;
	}

	SimpleMesh targetMesh;
	if (!targetMesh.loadMesh(filenameTarget)) {
		std::cout << "Mesh file wasn't read successfully." << std::endl;
		return;
	}

	PointCloud source{ sourceMesh };
	PointCloud target{ targetMesh };
	
	// Search for matches using FLANN.
	std::unique_ptr<NearestNeighborSearch> nearestNeighborSearch = std::make_unique<NearestNeighborSearchFlann>();
	nearestNeighborSearch->setMatchingMaxDistance(0.0001f);
	nearestNeighborSearch->buildIndex(target.getPoints());
	auto matches = nearestNeighborSearch->queryMatches(source.getPoints());

	// Visualize the correspondences with lines.
	SimpleMesh resultingMesh = SimpleMesh::joinMeshes(sourceMesh, targetMesh, Matrix4f::Identity());
	auto sourcePoints = source.getPoints();
	auto targetPoints = target.getPoints();

	for (unsigned i = 0; i < 100; ++i) { // sourcePoints.size()
		const auto match = matches[i];
		if (match.idx >= 0) {
			const auto& sourcePoint = sourcePoints[i];
			const auto& targetPoint = targetPoints[match.idx];
			resultingMesh = SimpleMesh::joinMeshes(SimpleMesh::cylinder(sourcePoint, targetPoint, 0.002f, 2, 15), resultingMesh, Matrix4f::Identity());
		}
	}

	resultingMesh.writeMesh(std::string("correspondences.off"));
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

int main() {
	// FaceModel model;

//    VectorXd random = VectorXd::Random(64)*2;
//    model.expCoefAr = random.data();
//    model.shapeCoefAr = random.data();
//
//    model.write_off("random.off");

    alignMeshWithICP();

    return 0;
}