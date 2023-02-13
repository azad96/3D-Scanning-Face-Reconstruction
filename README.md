# 3D Face Reconstruction from RGBD Images and Facial Expression Transfer
In this project, we aim to reconstruct 3D faces from RGB-D images and further transfer facial expressions among the reconstructed meshes. The initial task is to estimate the face parameters using non-linear optimization. We detect the facial landmarks of the face using an (external library) and optimize pose parameters using the Procrustres algorithm. We further learn the identity and optimize the shape parameters using ICP on depth values. These operations will be performed on both source and target images. Afterwards, the facial expressions of the source image will be transfered to the face model of the target image.


## Libraries
The following libraries need to be installed before running the project:

PCL 
Ceres 
glog
OpenCV
dlib 
glew 

## Configuring, building and running the project
mkdir build
cd build
cmake -DCMAKE_BUILD_TYPE=Release ..
make
./face_reconstruction