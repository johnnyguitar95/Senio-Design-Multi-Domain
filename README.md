Please refer to the orignal MDNet implementation. 

Our code provides a faster implementation using C++. 

Prereqs: 
gcc 4.8.5
OpenCV
Armadillo
MEX
(If necessary)
MEXOPENCV - https://www.mathworks.com/matlabcentral/fileexchange/47953-computer-vision-system-toolbox-opencv-interface

Setup - 
1. run compile_matconvnet
2. run setup_mdnet.m
3. mex utils/train_bbox_reg.cpp 
4. mex utils/predict_bbox_reg.cpp 
5. mexOpenCV or mex utils/im_crop.cpp

Running - 
You can configure which datsets to run through demo_tracking and genConfig
mdnet_run needs a list of paths to images, and initial bounding box as region, and a file path to a neural network for matconvnet

