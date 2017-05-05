#include <armadillo>
#include <opencv2/opencv.hpp>
#include "../utils/im_crop.cpp"
#include "mex.h"

using namespace arma;

/**
 * DOES NOT WORK PROPERLY
 */


cv::Mat* mdnet_extract_regions(cv::Ptr<cv::Mat> im, int boxes[][4], int num_boxes, int crop_mode, int crop_size, int crop_padding)
{
	//int num_boxes = size(boxes, 1);
	cv::Mat crop;
    int bbox[4] = {0,0,0,0};

	cv::Mat ims[num_boxes];// = malloc(sizeof(cv::Mat)*num_boxes);
// 	for(int x = 0; x < num_boxes; x++)
//         ims = Mat::zeros(crop_size, crop_size, CV_8UC3);

	for(int i = 0; i < num_boxes; i++)
	{
		memcpy(bbox, boxes[i], sizeof(bbox));
		//bbox = boxes.row(i); //extract row vector from boxes
        int mean_rgb[3] = {-1,-1,-1};
		crop = im_crop(im, bbox, crop_mode, crop_size, crop_padding, mean_rgb);
        ims[i] = crop;
        
// 		for(int i = 0; i < crop_size; i++)
// 			for(int j = 0; j < crop_size; j++)
// 				for(int k = 0; k < 2; k++)
// 					for(int l = 0; l < num_boxes; l++)
// 						ims(i, j,    
    
	}
    
    cv::Mat *ims2 = ims;
    
	return ims2;
}
plhs
void mexFunction(int nlhs, mxArray *plhs[], int nrhs, const mxArray *prhs[]) {

    
    cv::Ptr<cv::Mat> im = ocvMxArrayToImage_uint8(prhs[0], true);
	
    int num_boxes = *(mxGetPr(prhs[2]));
    
    int boxes[num_boxes][4]; 	//input for boxes
    double *bb = mxGetPr(prhs[1]);//one really long array
    
    int count = 0;
    for(int x = 0; x < num_boxes*4; x=x+4){
        for(int y = x; y < 4+x; y++){
            boxes[x][y] = *(bb+y);
        }
    }

	int crop_mode = *mxGetPr(prhs[3]);
    int crop_size = *mxGetPr(prhs[4]);
    int crop_padding = *mxGetPr(prhs[5]);
    
    cv::Mat *ims = mdnet_extract_regions(im, boxes, num_boxes, crop_mode, crop_size, crop_padding);
    
    for(int i = 0; i < num_boxes; i++){
        *(plhs[0])+i = ocvMxArrayFromImage_uint8(*ims+i);
    }
}

