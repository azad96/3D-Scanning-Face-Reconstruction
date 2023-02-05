#include <iostream>
#include <fstream>

#include "Eigen.h"
#include "VirtualSensor.h"
#include "SimpleMesh.h"
#include "ICPOptimizer.h"
#include "PointCloud.h"
#include "FaceModel.h"


template <typename T>
class PoseIncrement {
public:
    explicit PoseIncrement(T* const array) : m_array{ array } { }

    void setZero() {
        for (int i = 0; i < 6; ++i)
            m_array[i] = T(0);
    }

    T* getData() const {
        return m_array;
    }

    /**
     * Applies the pose increment onto the input point and produces transformed output point.
     * Important: The memory for both 3D points (input and output) needs to be reserved (i.e. on the stack)
     * beforehand).
     */
    void apply(T* inputPoint, T* outputPoint) const {
        // pose[0,1,2] is angle-axis rotation.
        // pose[3,4,5] is translation.
        const T* rotation = m_array;
        const T* translation = m_array + 3;

        T temp[3];
        ceres::AngleAxisRotatePoint(rotation, inputPoint, temp);

        outputPoint[0] = temp[0] + translation[0];
        outputPoint[1] = temp[1] + translation[1];
        outputPoint[2] = temp[2] + translation[2];
    }

    /**
     * Converts the pose increment with rotation in SO3 notation and translation as 3D vector into
     * transformation 4x4 matrix.
     */
    static MatrixXd convertToMatrix(const PoseIncrement<double>& poseIncrement) {


        // pose[0,1,2] is angle-axis rotation.
        // pose[3,4,5] is translation.
        double* pose = poseIncrement.getData();
        double* rotation = pose;
        double* translation = pose + 3;

        // Convert the rotation from SO3 to matrix notation (with column-major storage).
        double rotationMatrix[9];
        ceres::AngleAxisToRotationMatrix(rotation, rotationMatrix);

        // Create the 4x4 transformation matrix.
        MatrixXd matrix(3,4);
        // matrix.setIdentity();
        matrix(0, 0) = double(rotationMatrix[0]);	matrix(0, 1) = double(rotationMatrix[3]);	matrix(0, 2) = double(rotationMatrix[6]);	matrix(0, 3) = double(translation[0]);
        matrix(1, 0) = double(rotationMatrix[1]);	matrix(1, 1) = double(rotationMatrix[4]);	matrix(1, 2) = double(rotationMatrix[7]);	matrix(1, 3) = double(translation[1]);
        matrix(2, 0) = double(rotationMatrix[2]);	matrix(2, 1) = double(rotationMatrix[5]);	matrix(2, 2) = double(rotationMatrix[8]);	matrix(2, 3) = double(translation[2]);
        return matrix;
    }

private:
    T* m_array;
};

void debugCorrespondenceMatching() {
	// Load the source and target mesh.
	const std::string filenameSource = std::string("../sample_face/source_mesh.off");
	const std::string filenameTarget = std::string("../sample_face/target_mesh.off");

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

	resultingMesh.writeMesh(std::string("../sample_face/correspondences.off"));
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
	// double r[6] = {0, 3.14/9, 0, 0, 0, 0};
	// auto pose = PoseIncrement<double>(r);
	// auto matrix = PoseIncrement<double>::convertToMatrix(pose);


	
	// FaceModel* faceModel = FaceModel::getInstance();

	// VectorXd random = VectorXd::Random(64)*2;
	// faceModel->expCoefAr = random.data();
	// faceModel->shapeCoefAr = random.data();

	// auto mesh = faceModel->transform(matrix);
	// faceModel->write_off("../sample_face/rotated_random.off", mesh);

    alignMeshWithICP();

    return 0;
}