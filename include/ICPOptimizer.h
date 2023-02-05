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
static inline void dotFace_ceres(const T* expCoef, const T* shapeCoef, T* face) {
    FaceModel* faceModel = FaceModel::getInstance();
    T* expression =  new T[107127];
    T* shape = new T[107127];

    for (int i = 0; i < 107127; i++) {
        T sum = T(0.0);
        for (int j = 0; j < 64; j++) {
            sum += faceModel->expBaseAr[i][j] * expCoef[j];
        }
        expression[i] = T(sum);
    }

    for (int i = 0; i < 107127; i++) {
        T sum = T(0.0);
        for (int j = 0; j < 80; j++) {
            sum += faceModel->idBaseAr[i][j] * shapeCoef[j];
        }
        shape[i] = T(sum);
    }

    for (int i = 0; i < 107127; i++) {
        face[i] = expression[i] + shape[i] + faceModel->meanshapeAr[i];
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

    void apply( const int inputIndex, T* outputPoint) const {
        const T* expCoef = m_arrayExpCoef;
        const T* shapeCoef = m_arrayShapeCoef;
        T* faces = new T[107127];

        dotFace_ceres(expCoef, shapeCoef, faces);

        outputPoint[0] = faces[3*inputIndex + 0];
        outputPoint[1] = faces[3*inputIndex + 1];
        outputPoint[2] = faces[3*inputIndex + 2];
        delete faces;
    }

private:
    T* m_arrayExpCoef;
    T* m_arrayShapeCoef;
};


class MyCustomConstraint {
public:
    MyCustomConstraint(const int sourcePointIndex, const Vector3f& targetPoint, const float weight) :
            m_sourcePointIndex{ sourcePointIndex },
            m_targetPoint{ targetPoint },
            m_weight{ weight }
    { }

    template <typename T>
    bool operator()(const T* const expCoeff, const T* const shapeCoeff, T* residuals) const {
        auto expShapeCoeffIncrement = ExpShapeCoeffIncrement<T>(const_cast<T*>(expCoeff), const_cast<T*>(shapeCoeff));
        T p_s_tilda[3];
        expShapeCoeffIncrement.apply(m_sourcePointIndex, p_s_tilda);
        residuals[0] = T(LAMBDA) * T(m_weight) * (p_s_tilda[0] - T(m_targetPoint[0]));
        residuals[1] = T(LAMBDA) * T(m_weight) * (p_s_tilda[1] - T(m_targetPoint[1]));
        residuals[2] = T(LAMBDA) * T(m_weight) * (p_s_tilda[2] - T(m_targetPoint[2]));

        return true;
    }

    static ceres::CostFunction* create(const int sourcePointIndex, const Vector3f& targetPoint, const float weight) {
        return new ceres::AutoDiffCostFunction<MyCustomConstraint, 3, 64, 80>(
                new MyCustomConstraint(sourcePointIndex, targetPoint, weight)
        );
    }

protected:
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

    virtual void estimateExpShapeCoeffs(const PointCloud &target) override {
        // Build the index of the FLANN tree (for fast nearest neighbor lookup).
        m_nearestNeighborSearch->buildIndex(target.getPoints());

        // The initial estimate can be given as an argument.
        FaceModel* faceModel = FaceModel::getInstance();
        double *estimatedShapeCoef = faceModel->expCoefAr;

        double *estimatedExprCoef = faceModel->shapeCoefAr;

        double incrementArrayExp[64];
        double incrementArrayShape[80];

        auto expShapeCoeffIncrement = ExpShapeCoeffIncrement<double>(incrementArrayExp, incrementArrayShape);
        expShapeCoeffIncrement.setZero();

        for (int i = 0; i < m_nIterations; ++i) {
            // Compute the matches.
            std::cout << "Matching points ..." << std::endl;
            clock_t begin = clock();


            faceModel->write_off("transformed_model.off");
            SimpleMesh faceMesh;
            if (!faceMesh.loadMesh("transformed_model.off")) {
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
            customPrepareConstraints(target.getPoints(), matches, expShapeCoeffIncrement, problem);

            // Configure options for the solver.
            ceres::Solver::Options options;
            configureSolver(options);

            // Run the solver (for one iteration).
            ceres::Solver::Summary summary;
            ceres::Solve(options, &problem, &summary);
            std::cout << summary.BriefReport() << std::endl;
            //std::cout << summary.FullReport() << std::endl;

            //get updated optim. params
            estimatedExprCoef = expShapeCoeffIncrement.getExpCoeff();
            estimatedShapeCoef = expShapeCoeffIncrement.getShapeCoeff();

            //update face model with these params
            faceModel->expCoefAr = estimatedExprCoef;
            faceModel->shapeCoefAr = estimatedShapeCoef;

            std::cout << "Optimization iteration done." << std::endl;
        }
        faceModel->write_off("result.off");

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
                                  const std::vector<Match> matches,
                                  const ExpShapeCoeffIncrement<double> &expShapeCoeffIncrement,
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
                        expShapeCoeffIncrement.getShapeCoeff()
                );
            }
        }
    }
};