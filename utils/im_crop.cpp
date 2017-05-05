#include <mex.h>
#include "opencvmex.hpp"
#include <opencv2/opencv.hpp>
#include <stdlib.h>
#include <string>
#include <iostream>
#include <math.h>
#include <stddef.h>

/**
 * The most effective way to compile this file is with this tool
 * https://www.mathworks.com/matlabcentral/fileexchange/47953-computer-vision-system-toolbox-opencv-interface
 */


using namespace cv;
using namespace std;


string type2str(int type) {
  string r;

  uchar depth = type & CV_MAT_DEPTH_MASK;
  uchar chans = 1 + (type >> CV_CN_SHIFT);

  switch ( depth ) {
    case CV_8U:  r = "8U"; break;
    case CV_8S:  r = "8S"; break;
    case CV_16U: r = "16U"; break;
    case CV_16S: r = "16S"; break;
    case CV_32S: r = "32S"; break;
    case CV_32F: r = "32F"; break;
    case CV_64F: r = "64F"; break;
    default:     r = "User"; break;
  }

  r += "C";
  r += (chans+'0');

  return r;
}



/**
 * @param im - string for image file name
 * @param bbox - bounding box in order [x1, x2, y1, y2]
 * @param cropmode - integer for 'warp' or 'square', 1 and 2 respectively. 
 * @param padding - integer amount of padding to include at the target scale
 * @param mean_rgb - double for subtracting from the cropped window. Pass array with -1 to ignore
 */

cv::Mat im_crop(cv::Ptr<cv::Mat> im, int bbox[4], int crop_mode, int crop_size, int padding, int mean_rgb[3]){

    cv::Mat img = *im;
    
	bool use_square = false;
	// if crop_mode == 2, it's square)
	if (crop_mode == 2){
		use_square = true;
	}else if (crop_mode != 1){
		std::cout << "ERROR: im_crop, invalid value for crop_mode = " << crop_mode;
	}

	int pad_w = 0;
	int pad_h = 0;
	double crop_width = crop_size;
	double crop_height = crop_size;
    //good up to this point
	//working out dimensions to cropsingle
	if(padding > 0 || use_square){
        //std::cout << crop_size/(crop_size - (padding*2.0)) << "\n";
		double scale = crop_size/(crop_size - padding*2.0);
		double half_height = bbox[3]/2;
		double half_width = bbox[2]/2;
		double center[2] = {bbox[0]+half_width, bbox[1]+half_height};
		if (use_square){
			if (half_height > half_width)
				half_width = half_height;
			else
				half_height = half_width;
		}
        //std::cout << center[0] << " " << half_width << " " << scale << "\n";
        //std::cout << "center=[" << center[0] << "," << center[1] <<"]\n";
		bbox[0] = round(center[0] + ((half_width*-1))*scale);
        //std::cout << round(center[0] + ((half_width*-1))*scale) << "\n";
		bbox[1] = round(center[1] + ((half_height*-1))*scale);
		bbox[2] = round(center[0] + half_width*scale);
		bbox[3] = round(center[1] + half_height*scale);
        //std::cout << "BBOX=["<<bbox[0]<<","<<bbox[1]<<","<<bbox[2]<<","<<bbox[3]<<"]\n";
		double unclipped_height = bbox[3]-bbox[1]+1;
		double unclipped_width = bbox[2]-bbox[0]+1;
	
		double pad_x1 = 0; 
		if(1 - bbox[0] > 0)
			pad_x1 = 1 - bbox[0];
	
		double pad_y1 = 0;
		if(1 - bbox[1] > 0)
			pad_y1 = 1 - bbox[1];
		
		if(1 > bbox[0])
			bbox[0] = 1;
		if(1 > bbox[1])
			bbox[1] = 1;
		if(img.size().width < bbox[2])
			bbox[2] = img.size().width;
		if(img.size().height < bbox[3])
			bbox[3] = img.size().height;
        //std::cout << "BBOX=["<<bbox[0]<<","<<bbox[1]<<","<<bbox[2]<<","<<bbox[3]<<"]\n";
		double clipped_height = bbox[3] - bbox[1]+1;
		double clipped_width = bbox[2] - bbox[0]+1;
		double scale_x = crop_size/unclipped_width;
		double scale_y = crop_size/unclipped_height;
		crop_width = round(clipped_width*scale_x);
		crop_height = round(clipped_height*scale_y);
		pad_x1 = round(pad_x1*scale_x);
		pad_y1 = round(pad_y1*scale_y);

        pad_h = pad_y1;
        pad_w = pad_x1;
        
		if(pad_y1 + crop_height > crop_size)
			crop_height = crop_size - pad_y1;
		if(pad_x1 + crop_width > crop_size)
			crop_width = crop_size - pad_x1;
	}
    
	//this is the actual cropping of the image 	
	cv::Mat window;
	img(cv::Rect(bbox[0], bbox[1], bbox[2]-bbox[0], bbox[3]-bbox[1])).copyTo(window);
    
    /*cout << "Crop Height " << crop_height << "\n";
    cout << "Crop Width " << crop_width << "\n";*/
    crop_height = crop_size;
    crop_width = crop_size;
    
    cv::Size new_dimensions(crop_height, crop_width);
   
    cv::Mat tmp;
    cv::cvtColor(window, tmp, cv::COLOR_BGR2GRAY);
	resize(window, tmp, new_dimensions);
    /*cv::Mat sample_float;
    tmp.convertTo(sample_float, CV_32FC3);
    cv::Mat sample_normalized;
    cv::subtract(sample_float, */
    
    for(int y = 0; y<tmp.rows;y++){
        for(int x = 0; x<tmp.cols;x++){
            Vec3b color = tmp.at<Vec3b>(Point(x,y));
            if(mean_rgb[0] == -1){
                color[0] = color[0] - 128;
                color[1] = color[1] - 128;
                color[2] = color[2] - 128;
            }else{
                color[0] = color[0] - mean_rgb[0];
                color[1] = color[1] - mean_rgb[1];
                color[2] = color[2] - mean_rgb[2];
            }
            tmp.at<Vec3b>(Point(x,y)) = color;
        }//end x loop
    }//end y loop
    

   window = Mat::zeros(crop_size, crop_size, CV_8UC3);
   
   
   //tmp.convertTo(tmp, CV_8U);
   string ty1 =  type2str( window.type() );
   //std::cout << "Matrix: " << ty1.c_str() << " " << window.cols << "x" << window.rows <<"\n";
   
   string ty2 =  type2str( tmp.type() );
   //std::cout << "Matrix: " << ty2.c_str() << " " << tmp.cols << "x" << tmp.rows<<"\n";
   
   if(crop_mode==1){
       //std::cout << "Almost there\n";
       //tmp.copyTo(window(cv::Rect(pad_w,pad_h,pad_w+crop_width,pad_h+crop_height)));
       //std::cout << "FINISHED\n";
       return tmp;
   }
   else{
       //copy tmp into window. basically tmp should have a black border
       
       return window;
   }


   //std::cout << "Here\n";
   //window = Mat::zeros(crop_size, crop_size, CV_32F); 
   //tmp.copyTo(window(cv::Rect(0+pad_w, 0+pad_h, crop_width+pad_w, crop_height+pad_h)));
   //tmp.copyTo(window);
   /*
   for(int y = 0; y<tmp.rows;y++){
        for(int x = 0; x<tmp.cols;x++){
            Vec3b color = tmp.at<Vec3b>(Point(pad_h+y, pad_w+x));
            window.at<Vec3b>(Point(pad_h+crop_height, pad_w+crop_width)) = color;
        }
   }*/
   //return window;	
   
}

void mexFunction(int nlhs, mxArray *plhs[], int nrhs, const mxArray *prhs[]) {
    
    cv::Ptr<cv::Mat> im = ocvMxArrayToImage_uint8(prhs[0], true);
    //std::cout << prhs[1][1] << "\n";
    double *bb = mxGetPr(prhs[1]);
    //std::cout << *(bbox+2) << "\n";
    int bbox[4] = {0,0,0,0};
    for(int x=0;x<4;x++){
        bbox[x] = *(bb+x);
        //std::cout << bbox[x]<<"\n";
    }
    int crop_mode = 1;
    int crop_size = *(mxGetPr(prhs[3]));
    //std::cout << crop_size << "\n";
    int padding = *(mxGetPr(prhs[4]));
    //std::cout << padding << "\n";
    double *rgb = mxGetPr(prhs[5]);
    int mean_rgb[3] = {0,0,0};
    for(int x=0;x<3;x++){
        mean_rgb[x] = *(rgb+x);
        //std::cout << mean_rgb[x]<<"\n";
    }
    cv::Mat result;
    result = im_crop(im, bbox, crop_mode, crop_size, padding, mean_rgb);
    //cv::Mat result = *im;
    plhs[0] = ocvMxArrayFromImage_uint8(result);
    //std::cout << "This is c++\n";
}
