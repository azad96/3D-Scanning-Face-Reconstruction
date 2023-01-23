#include <iostream>
#include <fstream>
#include <vector>

#include "Eigen.h"
#include "FaceModel.h"



int main() 
{
	
	FaceModel model("C:/Users/adilm/Desktop/data");

	model.write_off(); 
	return 0;
}
