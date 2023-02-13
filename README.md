# 3D Face Reconstruction from RGBD Images and Facial Expression Transfer
In this project, we aim to reconstruct 3D faces from RGB-D images and further transfer facial expressions among the reconstructed meshes. The initial task is to estimate the face parameters using non-linear optimization. We detect the facial landmarks of the face using an (external library) and optimize pose parameters using the Procrustres algorithm. We further learn the identity and optimize the shape parameters using ICP on depth values. These operations will be performed on both source and target images. Afterwards, the facial expressions of the source image will be transfered to the face model of the target image.

## Libraries
The following libraries need to be installed before running the project:

~~~
PCL
FreeImage
Eigen
Flann
Ceres
glog
OpenCV
dlib
glew
~~~

## Dataset and Models
The IAS-Lab RGB-D Face Dataset needs to be downloaded.
The dlib detection model can be found at 
http://dlib.net/files/shape_predictor_68_face_landmarks.dat.bz2

It needs to be downloaded under the models directory.
models/shape_predictor_68_face_landmarks.dat

## Configuring, building and running the project
~~~
mkdir build
cd build
cmake -DCMAKE_BUILD_TYPE=Release ..
make
./face_reconstruction
~~~

To perform the expression transfer experiment, the macro parameter "TRANSFER" in the include/ICPOptimizer.h file must be changed. However, in order to be able to transfer, an experiment without transfer must be done first.

## Description of Files
All header files that main.cpp uses are inside the include directory.
### DataHandler.h
A class to read input image and point cloud from the dataset, and create the landmarks and a PointCloud class instance

### FaceMOdel.h 
A singleton class definition to hold all related face data and functions

### ICPOptimizer.h
Inside this header files, similar to exercises we have defined our constraints.

#### ShapeCostFunction
The struct defined for regularizing the shape parameter.

#### ExpressionCostFunction
The struct defined for regularizing the expression parameter.

#### dotFace_ceres
Function to evaluate the point coordinates of face model with given exp. and shape coeffs. for a certain index. Scaling, rotation and translation are applied here.

#### ExpShapeCoeffIncrement
Class to handle applying the transformation at current iteration with necessary parameters.

#### MyCustomConstraint
Class to be called by Ceres and added by AutoDiffCostFunction. Here, implements the point to point constraint.

#### PointToPlaneConstr
Class to be called by Ceres and added by AutoDiffCostFunction. Here, implements the point to plane constraint.
