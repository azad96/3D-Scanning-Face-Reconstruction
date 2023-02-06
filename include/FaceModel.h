#include <iostream>
#include <fstream>
#include <vector>

#pragma once

#include "Eigen.h"

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


static std::vector<unsigned int> readKeypoints(std::string fileToOpen) {

    std::vector<unsigned int> keypoints;

    std::string line;
    std::ifstream file(fileToOpen);
    std::string value;

    while (getline(file, line)) {

        std::stringstream lineStringStream(line);

        while (std::getline(lineStringStream, value, ','))
        {
            keypoints.push_back(stoi(value));
        }
    }
    return keypoints;
}


static std::vector<Triangle> readTriangle(std::string fileToOpen) {

    std::vector<unsigned int> triangle;
    std::vector<Triangle> m_triangles;

    std::string line;
    std::ifstream file(fileToOpen);
    std::string value;

    while (getline(file, line)) {

        Triangle t;
        std::stringstream lineStringStream(line);

        while (std::getline(lineStringStream, value, ','))
        {
            triangle.push_back(stoi(value));
        }

        t.idx0 = triangle[0];
        t.idx1 = triangle[1];
        t.idx2 = triangle[2];

        m_triangles.push_back(t);

        triangle.clear();
    }

    return m_triangles;
}


// Taken from https://github.com/AleksandarHaber/Save-and-Load-Eigen-Cpp-Matrices-Arrays-to-and-from-CSV-files/blob/master/source_file.cpp
static Eigen::MatrixXd readMatrixCsv(std::string fileToOpen)
{
    std::vector<double> matrixEntries;

    // in this object we store the data from the matrix
    std::ifstream matrixDataFile(fileToOpen);

    // this variable is used to store the row of the matrix that contains commas
    std::string matrixRowString;

    // this variable is used to store the matrix entry;
    std::string matrixEntry;

    // this variable is used to track the number of rows
    int matrixRowNumber = 0;


    while (getline(matrixDataFile, matrixRowString)) // here we read a row by row of matrixDataFile and store every line into the string variable matrixRowString
    {
        std::stringstream matrixRowStringStream(matrixRowString); //convert matrixRowString that is a string to a stream variable.

        while (std::getline(matrixRowStringStream, matrixEntry, ',')) // here we read pieces of the stream matrixRowStringStream until every comma, and store the resulting character into the matrixEntry
        {
            matrixEntries.push_back(stod(matrixEntry));   //here we convert the string to double and fill in the row vector storing all the matrix entries
        }
        matrixRowNumber++; //update the column numbers
    }

    return Map<Matrix<double, Dynamic, Dynamic, RowMajor>>(matrixEntries.data(), matrixRowNumber, matrixEntries.size() / matrixRowNumber);

}


class FaceModel {
protected:
    FaceModel() {

        std::cout << "Data reading..." << std::endl;
        std::string data_path = "../data";
        idBase = readMatrixCsv(data_path + "/idBase.csv");
        expBase = readMatrixCsv(data_path + "/expBase.csv");
        meanshape = readMatrixCsv(data_path + "/meanshape.csv");


        // key_points = readKeypoints(data_path + "/kp_inds.csv");
        m_triangles = readTriangle(data_path + "/tri.csv");

        std::cout << "Data reading completed..." << std::endl;

        meanshapeAr = new double[107127];
        expBaseAr = new double*[107127];
        idBaseAr = new double*[107127];

        for( int i = 0; i < 107127; i++) {
            meanshapeAr[i] = meanshape(i);

            expBaseAr[i] = new double[64];
            idBaseAr[i] = new double[80];

            for( int j = 0; j < 64; j++) {
                expBaseAr[i][j] = expBase(i, j);
            }

            for( int j = 0; j < 80; j++) {
                idBaseAr[i][j] = idBase(i, j);
            }
        }

        expCoefAr = new double[64];
        shapeCoefAr = new double[80];

        for( int j = 0; j < 64; j++) {
            expCoefAr[j] = 0.0;
        }

        for( int j = 0; j < 80; j++) {
            shapeCoefAr[j] = 0.0;
        }
        
        //Parameters for inner steps
        expression = new double[107127];
        shape = new double[107127];
        face = new double[107127];
        face_t = new double[107127];

        shapeCoef = VectorXd::Zero(80);
        expCoef = VectorXd::Zero(64);

        rotation = MatrixXd::Identity(3, 3);
        translation = VectorXd::Zero(3);
    }

    ~FaceModel() {
        for( int i = 0; i < 64; i++) {
            delete[] expBaseAr[i];
        }

        for( int i = 0; i < 80; i++) {
            delete[] idBaseAr[i];
        }

        delete[] expBaseAr;
        delete[] idBaseAr;
        delete[] meanshapeAr;

        delete[] expression;
        delete[] shape;
        delete[] face;
        delete[] face_t;

        delete[] shapeCoefAr;
        delete[] expCoefAr;
    }

public:
    FaceModel(FaceModel &other) = delete;

    void operator=(const FaceModel &) = delete;

    static FaceModel *getInstance();
    
    static double** getIdBaseAr();

    static double** getExpBaseAr();


    void clear() {
        shapeCoef = VectorXd::Zero(80);
        expCoef = VectorXd::Zero(64);

        rotation = MatrixXd::Identity(3, 3);
        translation = VectorXd::Zero(3);
    }

    double* get_mesh() {
        dotProduct(expBaseAr, expCoefAr, 64, expression);
        dotProduct(idBaseAr, shapeCoefAr, 80, shape);
        sum_params(expression, shape, meanshapeAr, face);
        return face;
    }


    Eigen::MatrixXd transform(Eigen::Matrix4d pose) {
        Eigen::MatrixXd mesh = getAsEigenMatrix(get_mesh());
        // return mesh * pose.block<3,4>(0,0);
        auto transposed_pose = pose.transpose();
        auto rotation = transposed_pose.block<3,3>(0,0);
        auto translation = transposed_pose.block<1,3>(3,0);

        return (mesh*rotation).rowwise() + translation;
    }

    Eigen::MatrixXd getAsEigenMatrix(double* face) {
        Eigen::MatrixXd result(35709, 3);
        for(int i = 0; i < 35709; i++) {
            for(int j = 0; j < 3; j++) {
                result(i,j) = face[i * 3 + j];
            }
        }
        return result;
    }

    void write_off(std::string filename) {
        std::cout << "Writing mesh...\n";
        std::ofstream file;

        Eigen::MatrixXd mesh = getAsEigenMatrix(get_mesh());
        
        file.open(filename.c_str());
        file << "OFF\n";
        file << "35709 70789 0\n";
        for (int i = 0; i < mesh.rows(); i++) {
            file << mesh(i, 0) << " " << mesh(i, 1) << " " << mesh(i, 2) << "\n";
        }
        for ( auto t : m_triangles) {
            file << "3 " << t.idx0 << " " << t.idx1 << " " << t.idx2 << "\n";
        }
    }

    void write_off(std::string filename, Eigen::MatrixXd mesh) {
        std::cout << "Writing mesh...\n";
        std::ofstream file;

        std::cout << mesh.rows() << " " << mesh.cols() << std::endl;

        file.open(filename.c_str());
        file << "OFF\n";
        file << "35709 70789 0\n";
        for (int i = 0; i < mesh.rows(); i++) {
            file << mesh(i, 0) << " " << mesh(i, 1) << " " << mesh(i, 2) << "\n";
        }
        for ( auto t : m_triangles) {
            file << "3 " << t.idx0 << " " << t.idx1 << " " << t.idx2 << "\n";
        }
    }


public:
    Eigen::MatrixXd idBase;
    Eigen::MatrixXd expBase;
    std::vector<Triangle> m_triangles;
    Eigen::MatrixXd meanshape;
    std::vector<unsigned int> key_points;
	std::vector<Eigen::Vector3d> vertices;
	Eigen::VectorXd faces;

    Eigen::VectorXd shapeCoef;
    Eigen::VectorXd expCoef;

	Eigen::MatrixXd rotation;
	Eigen::VectorXd translation;

	//Store in array for ceres usage
	double** idBaseAr;
	double** expBaseAr;
	double* meanshapeAr;
    double* shapeCoefAr;
    double* expCoefAr;

	//Parameters for inner steps
	double* expression;
    double* shape;
    double* face;
    double* face_t;

private:
    static FaceModel* m_pInstance;
    static std::mutex m_mutex;
};


FaceModel* FaceModel::m_pInstance{nullptr};
std::mutex FaceModel::m_mutex;


FaceModel *FaceModel::getInstance()
{
    std::lock_guard<std::mutex> lock(m_mutex);
    if (m_pInstance == nullptr)
    {
        m_pInstance = new FaceModel();
    }
    return m_pInstance;
}

double** FaceModel::getIdBaseAr()
{
    return FaceModel::getInstance()->idBaseAr;
}

double** FaceModel::getExpBaseAr()
{
    return FaceModel::getInstance()->expBaseAr;
}