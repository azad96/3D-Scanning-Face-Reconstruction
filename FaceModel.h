#include <iostream>
#include <fstream>
#include <vector>

#include "Eigen.h"

struct Vertex {
	EIGEN_MAKE_ALIGNED_OPERATOR_NEW

	// Position stored as 4 floats (4th component is supposed to be 1.0)
	Vector4f position;
	// Color stored as 4 unsigned char
	Vector4uc color;
};

struct Triangle {
	unsigned int idx0;
	unsigned int idx1;
	unsigned int idx2;

Triangle() : idx0{ 0 }, idx1{ 0 }, idx2{ 0 } {}

Triangle(unsigned int _idx0, unsigned int _idx1, unsigned int _idx2) :
	idx0(_idx0), idx1(_idx1), idx2(_idx2) {}
};

std::vector<unsigned int> readKeypoints(std::string fileToOpen) {

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

std::vector<Triangle> readTriangle(std::string fileToOpen) {

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
Eigen::MatrixXd readMatrixCsv(std::string fileToOpen)
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
public:
	FaceModel(std::string data_path) {

		std::cout << "Data reading..." << std::endl;

		idBase = readMatrixCsv(data_path + "/idBase.csv");
		expBase = readMatrixCsv(data_path + "/expBase.csv");
		meanshape = readMatrixCsv(data_path + "/meanshape.csv");

		std::cout << meanshape.rows() << " " << meanshape.cols() << std::endl;
		// key_points = readKeypoints(data_path + "/kp_inds.csv");
		m_triangles = readTriangle(data_path + "/tri.csv");

		std::cout << "Data reading completed..." << std::endl;

		shapeCoef = VectorXd::Zero(80);
		expCoef = VectorXd::Zero(64);

		rotation = MatrixXd::Identity(3, 3);
		translation = VectorXd::Zero(3);
    }
	
	void clear() {
		shapeCoef = VectorXd::Zero(80);
		expCoef = VectorXd::Zero(64);

		rotation = MatrixXd::Identity(3, 3);
		translation = VectorXd::Zero(3);
	}
	
	Eigen::MatrixXd get_mesh() {
		Eigen::MatrixXd face = (idBase * shapeCoef) +(expBase * expCoef) + meanshape;
		return face;
	}

	Eigen::MatrixXd transform(Eigen::MatrixXd face) {
		return face * rotation;
	}

	void write_off() {

		std::cout << "Writing mesh...\n";

		std::ofstream file;
		
		Eigen::MatrixXd mesh = get_mesh().reshaped<RowMajor>(35709, 3);

		std::cout << mesh.rows() << " " << mesh.cols() << std::endl;

		
		file.open("yeni.off");
		file << "OFF\n";
		file << "35709 70789 0\n";

		for (int i = 0; i < mesh.rows(); i++) {
			file << mesh(i, 0) << " " << mesh(i, 1) << " " << mesh(i, 2) << "\n";
		}

		for ( auto t : m_triangles) {
			file << "3 " << t.idx0 << " " << t.idx1 << " " << t.idx2 << "\n";
		}

	}

private:
    Eigen::MatrixXd idBase;
    Eigen::MatrixXd expBase;
    std::vector<Triangle> m_triangles;
    Eigen::MatrixXd meanshape;
    std::vector<unsigned int> key_points;

	Eigen::VectorXd shapeCoef;
	Eigen::VectorXd expCoef;

	Eigen::MatrixXd rotation;
	Eigen::VectorXd translation;
};