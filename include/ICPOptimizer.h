#pragma once

// The Google logging library (GLOG), used in Ceres, has a conflict with Windows defined constants. This definitions prevents GLOG to use the same constants
#define GLOG_NO_ABBREVIATED_SEVERITIES

#include <ceres/ceres.h>
#include <ceres/rotation.h>

#include "SimpleMesh.h"
#include "NearestNeighbor.h"
#include "PointCloud.h"
#include "ProcrustesAligner.h"
#include "FaceModel.h"


template <typename T>
static inline void dotFace_ceres(int pointIndex, const T* expCoef, const T* shapeCoef, T* face) {
    FaceModel* faceModel = FaceModel::getInstance();

    T* expression =  new T[3];
    T* shape = new T[3];

    for (int i = 0; i < 3; i++) {
        T sum = T(0.0);
        for (int j = 0; j < 64; j++) {
            sum += faceModel->expBaseAr[pointIndex*3 + i][j] * expCoef[j];
        }
        expression[i] = T(sum);
    }

    for (int i = 0; i < 3; i++) {
        T sum = T(0.0);
        for (int j = 0; j < 80; j++) {
            sum += faceModel->idBaseAr[pointIndex*3 + i][j] * shapeCoef[j];
        }
        shape[i] = T(sum);
    }

    for (int i = 0; i < 3; i++) {
        face[i] = expression[i] + shape[i] + faceModel->meanshapeAr[pointIndex*3 + i];
    }

    delete [] expression;
    delete [] shape;

}


template <typename T>
class ExpShapeCoeffIncrement {
public:
    explicit ExpShapeCoeffIncrement(T* const arrayExpCoef, T* const arrayShapeCoef) :
    m_arrayExpCoef{ arrayExpCoef },
    m_arrayShapeCoef{ arrayShapeCoef }
    { }

    void setZero() {
        for (int i = 0; i < 64; ++i)
            m_arrayExpCoef[i] = T(0);

        for (int i = 0; i < 80; ++i)
            m_arrayShapeCoef[i] = T(0);
    }

    T* getExpCoeff() const {
        return m_arrayExpCoef;
    }

    T* getShapeCoeff() const {
        return m_arrayShapeCoef;
    }

    void apply(const int inputIndex, T* outputPoint) const {
        const T* expCoef = m_arrayExpCoef;
        const T* shapeCoef = m_arrayShapeCoef;
        T* faces = new T[3];

        dotFace_ceres(inputIndex, expCoef, shapeCoef, faces);

        outputPoint[0] = faces[0];
        outputPoint[1] = faces[1];
        outputPoint[2] = faces[2];

        delete [] faces;
    }

private:
    T* m_arrayExpCoef;
    T* m_arrayShapeCoef;
};

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
    static Matrix4d convertToMatrix(const PoseIncrement<double>& poseIncrement) {
        // pose[0,1,2] is angle-axis rotation.
        // pose[3,4,5] is translation.
        double* pose = poseIncrement.getData();
        double* rotation = pose;
        double* translation = pose + 3;

        // Convert the rotation from SO3 to matrix notation (with column-major storage).
        double rotationMatrix[9];
        ceres::AngleAxisToRotationMatrix(rotation, rotationMatrix);

        // Create the 4x4 transformation matrix.
        Matrix4d matrix;
        matrix.setIdentity();
        matrix(0, 0) = double(rotationMatrix[0]);	matrix(0, 1) = double(rotationMatrix[3]);	matrix(0, 2) = double(rotationMatrix[6]);	matrix(0, 3) = double(translation[0]);
        matrix(1, 0) = double(rotationMatrix[1]);	matrix(1, 1) = double(rotationMatrix[4]);	matrix(1, 2) = double(rotationMatrix[7]);	matrix(1, 3) = double(translation[1]);
        matrix(2, 0) = double(rotationMatrix[2]);	matrix(2, 1) = double(rotationMatrix[5]);	matrix(2, 2) = double(rotationMatrix[8]);	matrix(2, 3) = double(translation[2]);

        return matrix;
    }

private:
    T* m_array;
};

class MyCustomConstraint {
public:
    MyCustomConstraint(const int sourcePointIndex, const Vector3f& targetPoint, const float weight) :
            m_sourcePointIndex{ sourcePointIndex },
            m_targetPoint{ targetPoint },
            m_weight{ weight }
    { }

    template <typename T>
    bool operator()(const T* const expCoeff, const T* const shapeCoeff, const T* const pose, T* residuals) const {
        auto expShapeCoeffIncrement = ExpShapeCoeffIncrement<T>(const_cast<T*>(expCoeff), const_cast<T*>(shapeCoeff));
        T p_s_tilda[3];
        expShapeCoeffIncrement.apply(m_sourcePointIndex, p_s_tilda);

        auto poseIncrement = PoseIncrement<T>(const_cast<T*>(pose));
        T transformedPoint[3];
        poseIncrement.apply(p_s_tilda, transformedPoint);

        residuals[0] = T(LAMBDA) * T(m_weight) * (transformedPoint[0] - T(m_targetPoint[0]));
        residuals[1] = T(LAMBDA) * T(m_weight) * (transformedPoint[1] - T(m_targetPoint[1]));
        residuals[2] = T(LAMBDA) * T(m_weight) * (transformedPoint[2] - T(m_targetPoint[2]));

        return true;
    }

    static ceres::CostFunction* create(const int sourcePointIndex, const Vector3f& targetPoint, const float weight) {
        return new ceres::AutoDiffCostFunction<MyCustomConstraint, 3, 64, 80, 6>(
                new MyCustomConstraint(sourcePointIndex, targetPoint, weight)
        );
    }

protected:
    const int m_sourcePointIndex;
    const Vector3f m_targetPoint;
    const float m_weight;
    const float LAMBDA = 0.1f;
};

class PointToPlaneConstr {
public:
    PointToPlaneConstr(const int sourcePointIndex, const Vector3f& targetPoint, const Vector3f& targetNormal, const float weight) :
            m_sourcePointIndex{ sourcePointIndex },
            m_targetPoint{ targetPoint },
            m_targetNormal{ targetNormal },
            m_weight{ weight }
    { }

    template <typename T>
    bool operator()(const T* const expCoeff, const T* const shapeCoeff, const T* const pose, T* residuals) const {
        auto expShapeCoeffIncrement = ExpShapeCoeffIncrement<T>(const_cast<T*>(expCoeff), const_cast<T*>(shapeCoeff));
        T p_s_tilda[3];
        expShapeCoeffIncrement.apply(m_sourcePointIndex, p_s_tilda);

        auto poseIncrement = PoseIncrement<T>(const_cast<T*>(pose));
        T transformedPoint[3];
        poseIncrement.apply(p_s_tilda, transformedPoint);

        residuals[0] =  T(LAMBDA) * T(m_weight) * T(m_targetNormal[0]) * (transformedPoint[0] - T(m_targetPoint[0]));
        residuals[0] += T(LAMBDA) * T(m_weight) * T(m_targetNormal[1]) * (transformedPoint[1] - T(m_targetPoint[1]));
        residuals[0] += T(LAMBDA) * T(m_weight) * T(m_targetNormal[2]) * (transformedPoint[2] - T(m_targetPoint[2]));

        return true;
    }

    static ceres::CostFunction* create(const int sourcePointIndex, const Vector3f& targetPoint, const Vector3f& targetNormal, const float weight) {
        return new ceres::AutoDiffCostFunction<PointToPlaneConstr, 1, 64, 80, 6>(
                new PointToPlaneConstr(sourcePointIndex, targetPoint, targetNormal, weight)
        );
    }

protected:
    const int m_sourcePointIndex;
    const Vector3f m_targetPoint;
    const Vector3f m_targetNormal;
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

    virtual void estimateExpShapeCoeffs(const PointCloud& target, Matrix4d& initialPose) = 0;


protected:
    unsigned m_nIterations;
    std::unique_ptr<NearestNeighborSearch> m_nearestNeighborSearch;

    void pruneCorrespondences(const std::vector<Vector3f>& sourceNormals, const std::vector<Vector3f>& targetNormals, std::vector<Match>& matches) {
        const unsigned nPoints = sourceNormals.size();

        for (unsigned i = 0; i < nPoints; i++) {
            Match& match = matches[i];
            if (match.idx >= 0) {
                const auto& sourceNormal = sourceNormals[i];
                const auto& targetNormal = targetNormals[match.idx];

                // TODO: Invalidate the match (set it to -1) if the angle between the normals is greater than 60
                if(acos(sourceNormal.dot(targetNormal) / (sourceNormal.norm() * targetNormal.norm())) * 180.0 / M_PI > 60.0) {
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

    virtual void estimateExpShapeCoeffs(const PointCloud &target, Matrix4d& initialPose) override {
        // Build the index of the FLANN tree (for fast nearest neighbor lookup).
        m_nearestNeighborSearch->buildIndex(target.getPoints());

        double incrementArrayExp[64];
        double incrementArrayShape[80];

        auto expShapeCoeffIncrement = ExpShapeCoeffIncrement<double>(incrementArrayExp, incrementArrayShape);
        expShapeCoeffIncrement.setZero();

        double incrementArray[6];
        auto poseIncrement = PoseIncrement<double>(incrementArray);
        poseIncrement.setZero();

        // The initial estimate can be given as an argument.
        Matrix4d estimatedPose = initialPose;

        FaceModel* faceModel = FaceModel::getInstance();
        for (int i = 0; i < m_nIterations; ++i) {
            // Compute the matches.
            std::cout << "Matching points ..." << std::endl;
            clock_t begin = clock();

            auto mesh = faceModel->transform(estimatedPose);
            faceModel->write_off("../sample_face/transformed_model.off", mesh);
            SimpleMesh faceMesh;
            if (!faceMesh.loadMesh("../sample_face/transformed_model.off")) {
                std::cout << "Mesh file wasn't read successfully at location: " << "transformed_model.off" << std::endl;
            }

            PointCloud faceModelPoints{faceMesh};
            auto matches = m_nearestNeighborSearch->queryMatches(faceModelPoints.getPoints());
            pruneCorrespondences(faceModelPoints.getNormals(), target.getNormals(), matches);

            clock_t end = clock();
            double elapsedSecs = double(end - begin) / CLOCKS_PER_SEC;
            std::cout << "Completed in " << elapsedSecs << " seconds." << std::endl;

            int matchCtr = 0;
            for(Match match : matches) {
                if (match.idx >= 0)
                   matchCtr++;
            }
            std::cout << "match count:" << matchCtr << std::endl;
            // Prepare point-to-point and point-to-plane constraints.
            ceres::Problem problem;
            customPrepareConstraints(target.getPoints(), target.getNormals(), matches, expShapeCoeffIncrement, poseIncrement, problem);

            // Configure options for the solver.
            ceres::Solver::Options options;
            configureSolver(options);

            // Run the solver (for one iteration).
            ceres::Solver::Summary summary;
            ceres::Solve(options, &problem, &summary);
            std::cout << summary.BriefReport() << std::endl;
            // std::cout << summary.FullReport() << std::endl;

            //update face model with these params
            faceModel->expCoefAr = expShapeCoeffIncrement.getExpCoeff();
            faceModel->shapeCoefAr = expShapeCoeffIncrement.getShapeCoeff();
            
            // Update the current pose estimate (we always update the pose from the left, using left-increment notation).
            Matrix4d matrix = PoseIncrement<double>::convertToMatrix(poseIncrement);
            estimatedPose = matrix * estimatedPose;
            poseIncrement.setZero();

            std::cout << "Optimization iteration done." << std::endl;
        }
        auto mesh = faceModel->transform(estimatedPose);
        faceModel->write_off("../sample_face/result.off", mesh);
    }


private:
    void configureSolver(ceres::Solver::Options &options) {
        // Ceres options.
        options.trust_region_strategy_type = ceres::LEVENBERG_MARQUARDT;
        options.use_nonmonotonic_steps = false;
        options.linear_solver_type = ceres::DENSE_QR;
        options.minimizer_progress_to_stdout = 1;
        options.max_num_iterations = 1;
        options.num_threads = 8;
    }

    void customPrepareConstraints(const std::vector<Vector3f> &targetPoints,
                                  const std::vector<Vector3f> &targetNormals,
                                  const std::vector<Match> matches,
                                  const ExpShapeCoeffIncrement<double> &expShapeCoeffIncrement,
                                  const PoseIncrement<double> &poseIncrement, 
                                  ceres::Problem &problem) const {
        const unsigned nPoints = targetPoints.size();
        for (unsigned i = 0; i < nPoints; ++i) {
            const auto match = matches[i];
            if (match.idx >= 0) {
                const auto &targetPoint = targetPoints[match.idx];

                const int sourcePointIndex = match.idx;
                problem.AddResidualBlock(
                        MyCustomConstraint::create(sourcePointIndex, targetPoint, match.weight),
                        nullptr,
                        expShapeCoeffIncrement.getExpCoeff(),
                        expShapeCoeffIncrement.getShapeCoeff(),
                        poseIncrement.getData()
                );

                const auto& targetNormal = targetNormals[match.idx];
                if (!targetNormal.allFinite())
                    continue;

                problem.AddResidualBlock(
                    PointToPlaneConstr::create(sourcePointIndex, targetPoint, targetNormal, match.weight),
                    nullptr, 
                    expShapeCoeffIncrement.getExpCoeff(),
                    expShapeCoeffIncrement.getShapeCoeff(),
                    poseIncrement.getData()
                );
            }
        }
    }
};