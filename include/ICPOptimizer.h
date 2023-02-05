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

#define NUM_VERTEX 35709

template <typename T>
static inline void dotFace_ceres(const T* idBaseRow, const T* expBaseRow, const T& meanShapeVal, const T* expCoef, const T* shapeCoef, T &face) {
    T sum = T(0.0);
    for (int j = 0; j < 64; j++) {
        sum += expBaseRow[j] * expCoef[j];
    }
    T expression = T(sum);

    sum = T(0.0);
    for (int j = 0; j < 80; j++) {
        sum += idBaseRow[j] * shapeCoef[j];
    }
    T shape = T(sum);

    face = expression + shape + meanShapeVal;
}



class MyCustomConstraint {
public:
    MyCustomConstraint(const FaceModel &faceModel, const int sourcePointIndex, const Vector3f& targetPoint, const float weight) :
            m_pFaceModel{ faceModel },
            m_sourcePointIndex{ sourcePointIndex },
            m_targetPoint{ targetPoint },
            m_weight{ weight }
    { }

    template <typename T>
    bool operator()(const T* const expCoeff, const T* const shapeCoeff, T* residuals) const {
        
        Eigen::Matrix<T, 3, 3> face;
        T p_s_tilda[3];

        //Shape loop
        for( int j = 0; j < 64; j++) {
            face(0,0) = T(m_pFaceModel.idBase(3*m_sourcePointIndex,j)) * shapeCoeff[j]; 
            face(0,1) = T(m_pFaceModel.idBase(3*m_sourcePointIndex+1,j)) * shapeCoeff[j];
            face(0,2) = T(m_pFaceModel.idBase(3*m_sourcePointIndex+2,j)) * shapeCoeff[j];
        }

        //Expression loop
        for( int j = 0; j < 64; j++) {
            face(1,0) = T(m_pFaceModel.expBase(3*m_sourcePointIndex,j)) * expCoeff[j]; 
            face(1,1) = T(m_pFaceModel.expBase(3*m_sourcePointIndex+1,j)) * expCoeff[j];
            face(1,2) = T(m_pFaceModel.expBase(3*m_sourcePointIndex+2,j)) * expCoeff[j];
        }

        face(2,0) = T(m_pFaceModel.meanshape(3*m_sourcePointIndex));
        face(2,1) = T(m_pFaceModel.meanshape(3*m_sourcePointIndex + 1));
        face(2,2) = T(m_pFaceModel.meanshape(3*m_sourcePointIndex + 2));


        for( int i = 0; i < 3; i++) {
            p_s_tilda[i] = face(0,i) + face(1,i) + face(2,i);
        }

        residuals[0] = T(LAMBDA) * T(m_weight) * (p_s_tilda[0] - T(m_targetPoint[0]));
        residuals[1] = T(LAMBDA) * T(m_weight) * (p_s_tilda[1] - T(m_targetPoint[1]));
        residuals[2] = T(LAMBDA) * T(m_weight) * (p_s_tilda[2] - T(m_targetPoint[2]));

        return true;
    }

    static ceres::CostFunction* create(const FaceModel &faceModel, const int sourcePointIndex, const Vector3f& targetPoint, const float weight) {
        return new ceres::AutoDiffCostFunction<MyCustomConstraint, 3, 64, 80>(
                new MyCustomConstraint(faceModel, sourcePointIndex, targetPoint, weight)
        );
    }

protected:
    const FaceModel m_pFaceModel;
    const int m_sourcePointIndex;
    const Vector3f m_targetPoint;
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

    virtual void estimateExpShapeCoeffs(const PointCloud& target) = 0;


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

    ///////////////////////////////////////////////////////////////////////////////////////////////////////////////////

    virtual void estimateExpShapeCoeffs(const PointCloud &target) override {

        // Build the index of the FLANN tree (for fast nearest neighbor lookup).
        m_nearestNeighborSearch->buildIndex(target.getPoints());
        
        Eigen::VectorXd idCoefParam = VectorXd::Zero(80);
        Eigen::VectorXd expCoefParam = VectorXd::Zero(64);

        for( int i = 0; i < 64; i++) {

        }

        for (int i = 0; i < m_nIterations; ++i) {
            // Compute the matches.
            std::cout << "Matching points ..." << std::endl;
            clock_t begin = clock();

            faceModel.write_off("../sample_face/transformed_model.off", 
                                idCoefParam.data(),
                                expCoefParam.data());

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
            customPrepareConstraints(target.getPoints(), matches, idCoefParam, expCoefParam, problem);

            // Configure options for the solver.
            ceres::Solver::Options options;
            configureSolver(options);

            // Run the solver (for one iteration).
            ceres::Solver::Summary summary;
            ceres::Solve(options, &problem, &summary);
            std::cout << summary.BriefReport() << std::endl;
            //std::cout << summary.FullReport() << std::endl;

            //update face model with these params
            // faceModel.update_face(estimatedShapeCoef, estimatedExprCoef);
            faceModel.write_off("../sample_face/transformed_model.off", 
                                idCoefParam.data(),
                                expCoefParam.data());

            std::cout << "Optimization iteration done." << std::endl;

        }
        std::cout << "2" << std::endl;
        faceModel.write_off("../sample_face/result.off",
                            idCoefParam.data(),
                            expCoefParam.data());
    }

private:
    FaceModel faceModel;

    void configureSolver(ceres::Solver::Options &options) {
        // Ceres options.
        options.trust_region_strategy_type = ceres::LEVENBERG_MARQUARDT;
        options.use_nonmonotonic_steps = false;
        options.linear_solver_type = ceres::DENSE_QR;
        options.minimizer_progress_to_stdout = 1;
        options.max_num_iterations = 1;
        options.num_threads = 1;
    }

    void customPrepareConstraints(const std::vector<Vector3f> &targetPoints,
                                  const std::vector<Match> matches,
                                  Eigen::VectorXd &idCoefParam,
                                  Eigen::VectorXd &expCoefParam,
                                  ceres::Problem &problem) const {
        const unsigned nPoints = targetPoints.size();
        for (unsigned i = 0; i < nPoints; ++i) {
            const auto match = matches[i];
            if (match.idx >= 0) {
                const auto &targetPoint = targetPoints[match.idx];

                const int sourcePointIndex = match.idx;
                problem.AddResidualBlock(
                        MyCustomConstraint::create(faceModel, sourcePointIndex, targetPoint, match.weight),
                        nullptr,
                        idCoefParam.data(),
                        expCoefParam.data());
            }
        }
    }
};