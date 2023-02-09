

#include <iostream>
#include <fstream>
#include <math.h>   


#include "VirtualSensor.h"
#include "SimpleMesh.h"
#include "ICPOptimizer.h"
#include "BridgeSparseDense.h"





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
    std::cout << "Hello World!" << std::endl;
    std::string source_filename = "procrusted_face.off";

    write_prcrusted_mesh(source_filename);

    alignMeshWithICP();

    return 0;
}