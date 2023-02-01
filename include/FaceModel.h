#include <iostream>
#include <fstream>
#include <vector>

#include "Eigen.h"


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

		m_idBase = readMatrixCsv(data_path + "/idBase.csv");
		m_expBase = readMatrixCsv(data_path + "/expBase.csv");
		m_meanShape = readMatrixCsv(data_path + "/meanshape.csv");

		std::cout << m_meanShape.rows() << " " << m_meanShape.cols() << std::endl;
		m_keyPoints = readKeypoints(data_path + "/kp_inds.csv");
		
		std::cout<<m_keyPoints.size()<<std::endl;
		m_triangles = readTriangle(data_path + "/tri.csv");

		std::cout << "Data reading completed..." << std::endl;

		m_shapeCoef = VectorXd::Zero(80);
		m_expCoef = VectorXd::Zero(64);

		m_rotation = MatrixXd::Identity(3, 3);
		m_translation = VectorXd::Zero(3);

		updateMesh();

		createKeyVector();
    }
	
	void clear() {
		m_shapeCoef = VectorXd::Zero(80);
		m_expCoef = VectorXd::Zero(64);

		m_rotation = MatrixXd::Identity(3, 3);
		m_translation = VectorXd::Zero(3);
	}

	void initializeRandomExpression() {
		m_expCoef = VectorXd::Random(64)*2;
		
	}

	void setRotation(MatrixXd rotation) {
		m_rotation = rotation;
	}

	void setTranslation(VectorXd translation) {
		m_translation = translation;
	}

	void createKeyVector(){
		for(int i = 0 ; i < m_keyPoints.size() ; i++){
			Vector3f a(m_face(m_keyPoints[i], 0), m_face(m_keyPoints[i], 1), m_face(m_keyPoints[i], 2));
			m_keyVectors.push_back(a);
		}
	}
	
	void updateMesh() {
		m_face = (m_idBase * m_shapeCoef) +(m_expBase * m_expCoef) + m_meanShape;
		m_face = m_face.reshaped<RowMajor>(35709, 3);

	}

	void transform() {
		int N = m_face.rows();
		std::cout << m_face.rows() << " " << m_face.cols() << std::endl;

		Eigen::Matrix4d Trans; 
		Trans.setIdentity();   
		Trans.block<3,3>(0,0) = m_rotation;
		Trans.block<3,1>(0,3) = m_translation;

		Eigen::VectorXd oneColumn= VectorXd::Zero(N);
		oneColumn.setOnes();
		m_face.conservativeResize(N, 4);
		m_face.col(3) = oneColumn;

		m_face = (m_face * Trans).block(0, 0, N, 3);
		std::cout << m_face.rows() << " " << m_face.cols() << std::endl;
		
		// m_face = (m_rotation * m_face.transpose()).colwise() + m_translation;
		// m_face = m_face.transposeInPlace();
	}

	void write_off(std::string filename) {
		std::cout << "Writing mesh...\n";

		std::ofstream file;
		
		Eigen::MatrixXd mesh = m_face.reshaped<RowMajor>(35709, 3);

		std::cout << mesh.rows() << " " << mesh.cols() << std::endl;

		
		file.open(filename.c_str());
		file << "OFF\n";
		file << "35709 70789 0\n";

		for (int i = 0; i < mesh.rows(); i++) 
			file << mesh(i, 0) << " " << mesh(i, 1) << " " << mesh(i, 2) << "\n";

		for ( auto t : m_triangles) 
			file << "3 " << t.idx0 << " " << t.idx1 << " " << t.idx2 << "\n";
		
	}


public:
	std::vector<Vector3f> m_keyVectors;

private:
    Eigen::MatrixXd m_idBase;
    Eigen::MatrixXd m_expBase;
    std::vector<Triangle> m_triangles;
    Eigen::MatrixXd m_meanShape;
    std::vector<unsigned int> m_keyPoints;

	Eigen::VectorXd m_shapeCoef;
	Eigen::VectorXd m_expCoef;

	Eigen::MatrixXd m_rotation;
	Eigen::VectorXd m_translation;

	Eigen::MatrixXd m_face;
};