#pragma once
#include "SimpleMesh.h"

class ProcrustesAligner {
public:
	Matrix4f estimatePose(const std::vector<Vector3f>& oldSourcePoints, const std::vector<Vector3f>& oldTargetPoints) {

		//TODO : why it is Nan ????????
        // first I removed nans
        std::vector<Vector3f> sourcePoints ;
        std::vector<Vector3f> targetPoints ;

        ASSERT(oldSourcePoints.size() == oldTargetPoints.size() && "The number of source and target points should be the same, since every source point is matched with corresponding target point.");


        for(int i=0 ; i<oldSourcePoints.size(); i++){
            if (std::isnan(oldSourcePoints[i][0]) || std::isnan(oldTargetPoints[i][0])||
                std::isnan(oldSourcePoints[i][1]) || std::isnan(oldTargetPoints[i][1])||
                std::isnan(oldSourcePoints[i][2]) || std::isnan(oldTargetPoints[i][2]) ){
				continue;
			}
            else{
                sourcePoints.push_back(oldSourcePoints[i]);
                targetPoints.push_back(oldTargetPoints[i]);
            }
        }




        // We estimate the pose between source and target points using Procrustes algorithm.
		// Our shapes have the same scale, therefore we don't estimate scale. We estimated rotation and translation
		// from source points to target points.


        
        ASSERT(sourcePoints.size() == targetPoints.size() && "The number of source and target points should be the same, since every source point is matched with corresponding target point.");


        


		auto sourceMean = computeMean(sourcePoints);
		auto targetMean = computeMean(targetPoints);

        
		
		Matrix3f rotation = estimateRotation(sourcePoints, sourceMean, targetPoints, targetMean);
		Vector3f translation = computeTranslation(sourceMean, targetMean);

		// To apply the pose to point x on shape X in the case of Procrustes, we execute:
		// 1. Translation of a point to the shape Y: x' = x + t
		// 2. Rotation of the point around the mean of shape Y: 
		//    y = R (x' - yMean) + yMean = R (x + t - yMean) + yMean = R x + (R t - R yMean + yMean)
		
		Matrix4f estimatedPose = Matrix4f::Identity();
		estimatedPose.block(0, 0, 3, 3) = rotation;
		estimatedPose.block(0, 3, 3, 1) = rotation * translation - rotation * targetMean + targetMean;

		return estimatedPose;
	}
	Vector3f computeMean(const std::vector<Vector3f>& points) {
		// Compute the mean of input points.
		const unsigned nPoints = points.size();
		int s = 0;
		Vector3f mean = Vector3f::Zero();
		for (int i = 0; i < nPoints; ++i) {
			if(!isnan(points[i](0))){
				mean += points[i];
				s++;
			}
		}
		mean /= s;
		return mean;
	}

private:
	

	Matrix3f estimateRotation(const std::vector<Vector3f>& sourcePoints, const Vector3f& sourceMean, const std::vector<Vector3f>& targetPoints, const Vector3f& targetMean) {
		// Estimate the rotation from source to target points, following the Procrustes algorithm. 
		// To compute the singular value decomposition you can use JacobiSVD() from Eigen.
		const unsigned nPoints = sourcePoints.size();
		MatrixXf sourceMatrix(nPoints, 3);
		MatrixXf targetMatrix(nPoints, 3);

		for (int i = 0; i < nPoints; ++i) {
			sourceMatrix.block(i, 0, 1, 3) = (sourcePoints[i] - sourceMean).transpose();
			targetMatrix.block(i, 0, 1, 3) = (targetPoints[i] - targetMean).transpose();
		}

		Matrix3f A = targetMatrix.transpose() * sourceMatrix;
		JacobiSVD<Matrix3f> svd(A, ComputeFullU | ComputeFullV);
		const Matrix3f& U = svd.matrixU();
		const Matrix3f& V = svd.matrixV();
	
		const float d = (U * V.transpose()).determinant();
		Matrix3f D = Matrix3f::Identity();
		D(2, 2) = d;
		
		Matrix3f R = U * D * V.transpose(); // the multiplication by D is necessary since UV' is only orthogonal, but not necessarily a rotation matrix
		return R;
	}

	Vector3f computeTranslation(const Vector3f& sourceMean, const Vector3f& targetMean) {
		// Compute the translation vector from source to target points.
		Vector3f translation = targetMean - sourceMean;
		return translation;
	}
};