#pragma once

// The Google logging library (GLOG), used in Ceres, has a conflict with Windows defined constants. This definitions prevents GLOG to use the same constants
#define GLOG_NO_ABBREVIATED_SEVERITIES

#include <ceres/ceres.h>
#include <ceres/rotation.h>
#include <flann/flann.hpp>

#include "SimpleMesh.h"
#include "NearestNeighbor.h"
#include "PointCloud.h"
#include "ProcrustesAligner.h"
#include "FaceModel.h"


template <typename T>
static inline void dotProduct(double** input1, double* input2, const int dim, T* output) {
	for (int i = 0; i < 107127; i++) {
		double sum = 0;
		for (int j = 0; j < dim; j++) {
			sum += input1[i][j] * input2[j];
		}
		output[i] = T(sum);
	}
}

template <typename T>
static inline void sum_params(double* input1, double* input2, double* input3, T* output) {

    for (int i = 0; i < 107127; i++) {
		output[i] = T(input1[i]) + T(input2[i]) + T(input3[i]);
	}
}


/**
 * Helper methods for writing Ceres cost functions.
 */
template <typename T>
static inline void fillVector(const Vector3f& input, T* output) {
    output[0] = T(input[0]);
    output[1] = T(input[1]);
    output[2] = T(input[2]);
}


/**
 * Pose increment is only an interface to the underlying array (in constructor, no copy
 * of the input array is made).
 * Important: Input array needs to have a size of at least 6.
 */
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
        int num_vertices = 35709;
        const T* rotation = m_array;
        const T* translation = m_array + 3;

        for( int i = 0; i < num_vertices; i++) {
            
            T point[3];

            point[0] = inputPoint[3*i];
            point[1] = inputPoint[3*i+1];
            point[2] = inputPoint[3*i+2];

            T temp[3];
            ceres::AngleAxisRotatePoint(rotation, inputPoint, temp);

            outputPoint[3*i] = temp[0] + translation[0];
            outputPoint[3*i+1] = temp[1] + translation[1];
            outputPoint[3*i+2] = temp[2] + translation[2];
        }
    }

    static Matrix4d convert() {
        // pose[0,1,2] is angle-axis rotation.
        // pose[3,4,5] is translation.
        
        double* pose = m_array;
        double* rotation = pose;
        double* translation = pose + 3;

        // Convert the rotation from SO3 to matrix notation (with column-major storage).
        double rotationMatrix[9];
        ceres::AngleAxisToRotationMatrix(rotation, rotationMatrix);

        // Create the 4x4 transformation matrix.
        Matrix4f matrix;
        matrix.setIdentity();
        matrix(0, 0) = float(rotationMatrix[0]);	matrix(0, 1) = float(rotationMatrix[3]);	matrix(0, 2) = float(rotationMatrix[6]);	matrix(0, 3) = float(translation[0]);
        matrix(1, 0) = float(rotationMatrix[1]);	matrix(1, 1) = float(rotationMatrix[4]);	matrix(1, 2) = float(rotationMatrix[7]);	matrix(1, 3) = float(translation[1]);
        matrix(2, 0) = float(rotationMatrix[2]);	matrix(2, 1) = float(rotationMatrix[5]);	matrix(2, 2) = float(rotationMatrix[8]);	matrix(2, 3) = float(translation[2]);
        matrix(3, 0) = float(0);                    matrix(3, 1) = float(0);                    matrix(3, 2) = float(0);                    matrix(3, 3) = float(1.0);
        return matrix;
    }

    /**
     * Converts the pose increment with rotation in SO3 notation and translation as 3D vector into
     * transformation 4x4 matrix.
     */
    static Matrix4f convertToMatrix(const PoseIncrement<double>& poseIncrement) {
        // pose[0,1,2] is angle-axis rotation.
        // pose[3,4,5] is translation.
        double* pose = poseIncrement.getData();
        double* rotation = pose;
        double* translation = pose + 3;

        // Convert the rotation from SO3 to matrix notation (with column-major storage).
        double rotationMatrix[9];
        ceres::AngleAxisToRotationMatrix(rotation, rotationMatrix);

        // Create the 4x4 transformation matrix.
        Matrix4f matrix;
        matrix.setIdentity();
        matrix(0, 0) = float(rotationMatrix[0]);	matrix(0, 1) = float(rotationMatrix[3]);	matrix(0, 2) = float(rotationMatrix[6]);	matrix(0, 3) = float(translation[0]);
        matrix(1, 0) = float(rotationMatrix[1]);	matrix(1, 1) = float(rotationMatrix[4]);	matrix(1, 2) = float(rotationMatrix[7]);	matrix(1, 3) = float(translation[1]);
        matrix(2, 0) = float(rotationMatrix[2]);	matrix(2, 1) = float(rotationMatrix[5]);	matrix(2, 2) = float(rotationMatrix[8]);	matrix(2, 3) = float(translation[2]);
        matrix(3, 0) = float(0);                    matrix(3, 1) = float(0);                    matrix(3, 2) = float(0);                    matrix(3, 3) = float(1.0);
        return matrix;
    }

private:
    T* m_array;
};


/**
 * Optimization constraints.
 */
class PointsConstraint {
public:
    PointsConstraint(const double** idBase, const double** expBase, const double* meanshape, const std::vector<Eigen::Vector3f> matches, const float weight) :    
        m_idBase{ idBase},
        m_expBase{ expBase},
        m_meanshape{ meanshape},
        m_matches{ matches},
        m_weight{ weight }
    { }

    template <typename T>
    bool operator()(const T* const pose, const T* const expCoef, const T* const shapeCoef, T* residuals) const {
        
        int num_vertices = 35709;
        T* expression = new T[num_vertices * 3];
        T* shape = new T[num_vertices * 3];
        T* face = new T[num_vertices * 3];
        T* face_t = new T[num_vertices * 3];

        //Create face from parameters
        dotProduct(m_expBase, expCoef, 64, expression);
        dotProduct(m_idBase, shapeCoef, 80, shape);
        sum_params(expression, shape, m_meanshape, face);

        auto poseIncrement = PoseIncrement<T>(const_cast<T*>(pose));
        
        poseIncrement.apply(face, face_t);
        
        //Residual calculation
        for( int i = 0; i < num_vertices; i++) {
            const auto& match = m_matches[i];
            if(match.idx >= 0) {
                //determine the target point and create array
                T targetPoint[3];
                fillVector(targetPoints[match.idx], targetPoint);

                residuals[i*3] = T(LAMBDA) * T(m_weight) * (face_t[i*3] - T(targetPoint[0]));
                residuals[i*3 + 1] = T(LAMBDA) * T(m_weight) * (face_t[i*3 + 1] - T(targetPoint[1]));
                residuals[i*3 + 2] = T(LAMBDA) * T(m_weight) * (face_t[i*3 + 2] - T(targetPoint[2]));
            }
        }

        return true;
    }

    static ceres::CostFunction* create(const double** idBase, const double** expBase, const double* meanshape, const std::vector<Eigen::Vector3f> matches, const float weight) {
        return new ceres::AutoDiffCostFunction<PointsConstraint, 107127, 6, 64, 80>(
            new PointsConstraint(idBase, expBase, meanshape, matches, weight)
            );
    }

protected:
    const std::vector<Eigen::Vector3f> m_matches;
    const double** m_idBase;
    const double** m_expBase;
    const double* m_meanshape;
    const float m_weight;
    const float LAMBDA = 0.1f;
};

/**
 * Optimization constraints.
 */
class PointsModelConstraint {
public:
    PointsModelConstraint(const FaceModel* model, const std::vector<Eigen::Vector3f> matches, const float weight) :  
        m_model{ model},
        m_matches{ matches},
        m_weight{ weight }
    { }

    template <typename T>
    bool operator()(const T* const pose, const T* const expCoef, const T* const shapeCoef, T* residuals) const {
        
        int num_vertices = 35709;
        T* expression = m_model-> expression;
        T* shape = m_model->shape;
        T* face = m_model->face;
        T* face_t = m_model->face_t;

        //Create face from parameters
        dotProduct(m_model->expBaseAr, expCoef, 64, expression);
        dotProduct(m_model->shape, shapeCoef, 80, shape);
        sum_params(expression, shape, m_model->meanshapeAr, face);

        auto poseIncrement = PoseIncrement<T>(const_cast<T*>(pose));
        
        poseIncrement.apply(face, face_t);
        
        //Residual calculation
        for( int i = 0; i < num_vertices; i++) {
            const auto& match = m_matches[i];
            if(match.idx >= 0) {
                //determine the target point and create array
                T targetPoint[3];
                fillVector(targetPoints[match.idx], targetPoint);

                residuals[i*3] = T(LAMBDA) * T(m_weight) * (face_t[i*3] - T(targetPoint[0]));
                residuals[i*3 + 1] = T(LAMBDA) * T(m_weight) * (face_t[i*3 + 1] - T(targetPoint[1]));
                residuals[i*3 + 2] = T(LAMBDA) * T(m_weight) * (face_t[i*3 + 2] - T(targetPoint[2]));
            }
        }

        return true;
    }

    static ceres::CostFunction* create(const FaceModel* model, const std::vector<Eigen::Vector3f> matches, const float weight) {
        return new ceres::AutoDiffCostFunction<PointsModelConstraint, 107127, 6, 64, 80>(
            new PointsModelConstraint(model, matches, weight)
            );
    }

protected:
    const std::vector<Eigen::Vector3f> m_matches;
    const FaceModel* m_model;
    const float m_weight;
    const float LAMBDA = 0.1f;
};


/**
 * ICP optimizer - Abstract Base Class
 */
class ICPOptimizer {
public:
    ICPOptimizer() :
        m_nIterations{ 20 },
        m_nearestNeighborSearch{ std::make_unique<NearestNeighborSearchFlann>() }
    { }

    void setMatchingMaxDistance(float maxDistance) {
        m_nearestNeighborSearch->setMatchingMaxDistance(maxDistance);
    }

    void setNbOfIterations(unsigned nIterations) {
        m_nIterations = nIterations;
    }

    virtual void estimatePose(const PointCloud& source, const PointCloud& target, Matrix4f& initialPose) = 0;

protected:
    unsigned m_nIterations;
    std::unique_ptr<NearestNeighborSearch> m_nearestNeighborSearch;

    std::vector<Vector3f> transformPoints(const std::vector<Vector3f>& sourcePoints, const Matrix4f& pose) {
        std::vector<Vector3f> transformedPoints;
        transformedPoints.reserve(sourcePoints.size());

        const auto rotation = pose.block(0, 0, 3, 3);
        const auto translation = pose.block(0, 3, 3, 1);

        for (const auto& point : sourcePoints) {
            transformedPoints.push_back(rotation * point + translation);
        }

        return transformedPoints;
    }

    std::vector<Vector3f> transformNormals(const std::vector<Vector3f>& sourceNormals, const Matrix4f& pose) {
        std::vector<Vector3f> transformedNormals;
        transformedNormals.reserve(sourceNormals.size());

        const auto rotation = pose.block(0, 0, 3, 3);

        for (const auto& normal : sourceNormals) {
            transformedNormals.push_back(rotation.inverse().transpose() * normal);
        }

        return transformedNormals;
    }

    void pruneCorrespondences(const std::vector<Vector3f>& sourceNormals, const std::vector<Vector3f>& targetNormals, std::vector<Match>& matches) {
        const unsigned nPoints = sourceNormals.size();

        for (unsigned i = 0; i < nPoints; i++) {
            Match& match = matches[i];
            if (match.idx >= 0) {
                const auto& sourceNormal = sourceNormals[i];
                const auto& targetNormal = targetNormals[match.idx];

                // TODO: Invalidate the match (set it to -1) if the angle between the normals is greater than 60
                double radian = std::acos(sourceNormal.dot(targetNormal) / (sourceNormal.norm() * targetNormal.norm()));
                double angle = radian * 180. / M_PI;
                if (angle > 60.) {
                    match.idx = -1;
                }
            }
        }
    }
};


/**
 * ICP optimizer - using Ceres for optimization.
 */
class CeresICPOptimizer : public ICPOptimizer {
public:
    CeresICPOptimizer() {}

    virtual void estimatePose(const PointCloud& source, const PointCloud& target, Matrix4f& initialPose) override {
        // Build the index of the FLANN tree (for fast nearest neighbor lookup).
        m_nearestNeighborSearch->buildIndex(target.getPoints());

        // The initial estimate can be given as an argument.
        Matrix4f estimatedPose = initialPose;

        // We optimize on the transformation in SE3 notation: 3 parameters for the axis-angle vector of the rotation (its length presents
        // the rotation angle) and 3 parameters for the translation vector. 
        double incrementArray[6];
        auto poseIncrement = PoseIncrement<double>(incrementArray);
        poseIncrement.setZero();

        for (int i = 0; i < m_nIterations; ++i) {
            // Compute the matches.
            std::cout << "Matching points ..." << std::endl;
            clock_t begin = clock();

            auto transformedPoints = transformPoints(source.getPoints(), estimatedPose);
            auto transformedNormals = transformNormals(source.getNormals(), estimatedPose);

            auto matches = m_nearestNeighborSearch->queryMatches(transformedPoints);
            pruneCorrespondences(transformedNormals, target.getNormals(), matches);

            clock_t end = clock();
            double elapsedSecs = double(end - begin) / CLOCKS_PER_SEC;
            std::cout << "Completed in " << elapsedSecs << " seconds." << std::endl;

            // Prepare point-to-point and point-to-plane constraints.
            ceres::Problem problem;
            prepareConstraints(transformedPoints, target.getPoints(), target.getNormals(), matches, poseIncrement, problem);

            // Configure options for the solver.
            ceres::Solver::Options options;
            configureSolver(options);

            // Run the solver (for one iteration).
            ceres::Solver::Summary summary;
            ceres::Solve(options, &problem, &summary);
            std::cout << summary.BriefReport() << std::endl;
            //std::cout << summary.FullReport() << std::endl;

            // Update the current pose estimate (we always update the pose from the left, using left-increment notation).
            Matrix4f matrix = PoseIncrement<double>::convertToMatrix(poseIncrement);
            estimatedPose = PoseIncrement<double>::convertToMatrix(poseIncrement) * estimatedPose;
            poseIncrement.setZero();

            std::cout << "Optimization iteration done." << std::endl;
        }

        // Store result
        initialPose = estimatedPose;
    }


private:
    void configureSolver(ceres::Solver::Options& options) {
        // Ceres options.
        options.trust_region_strategy_type = ceres::LEVENBERG_MARQUARDT;
        options.use_nonmonotonic_steps = false;
        options.linear_solver_type = ceres::DENSE_QR;
        options.minimizer_progress_to_stdout = 1;
        options.max_num_iterations = 1;
        options.num_threads = 8;
    }

    void prepareConstraints(const std::vector<Vector3f>& sourcePoints, const std::vector<Vector3f>& targetPoints, const std::vector<Vector3f>& targetNormals, const std::vector<Match> matches, const PoseIncrement<double>& poseIncrement, ceres::Problem& problem) const {
        const unsigned nPoints = sourcePoints.size();

        for (unsigned i = 0; i < nPoints; ++i) {
            const auto match = matches[i];
            if (match.idx >= 0) {
                const auto& sourcePoint = sourcePoints[i];
                const auto& targetPoint = targetPoints[match.idx];

                if (!sourcePoint.allFinite() || !targetPoint.allFinite())
                    continue;

                // TODO: Create a new point-to-point cost function and add it as constraint (i.e. residual block) 
                // to the Ceres problem.
                
                problem.AddResidualBlock(
                    new ceres::AutoDiffCostFunction<PointToPointConstraint, 3, 6>(
                        new PointToPointConstraint(sourcePoint, targetPoint, match.weight)
                        ),
                    nullptr, poseIncrement.getData()
                );

                if (m_bUsePointToPlaneConstraints) {
                    const auto& targetNormal = targetNormals[match.idx];

                    if (!targetNormal.allFinite())
                        continue;

                    // TODO: Create a new point-to-plane cost function and add it as constraint (i.e. residual block) 
                    // to the Ceres problem.
                    problem.AddResidualBlock(
                    new ceres::AutoDiffCostFunction<PointToPlaneConstraint, 1, 6>(
                        new PointToPlaneConstraint(sourcePoint, targetPoint, targetNormal, match.weight)
                        ),
                    nullptr, poseIncrement.getData()
                );

                }
            }
        }
    }
};
