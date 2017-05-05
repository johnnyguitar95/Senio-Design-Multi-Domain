#include <mex.h>
#include <iostream>
#include <armadillo>
#include <math.h>

using namespace arma;

arma::mat overlap_ratio(arma::mat rect1, arma::mat rect2);

arma::mat overlap_ratio(arma::mat rect1, arma::mat rect2){

    arma::mat inter_area = mat(rect1.n_rows, 1);
    arma::mat union_area = mat(rect1.n_rows, 1);
    
    for(int i = 0; i<inter_area.n_rows; i++){
        //computing overlap
        
        double dx = std::min(rect1(i,0)+rect1(i,2), rect2(i,0)+rect2(i,2)) - std::max(rect1(i,0),rect2(i,0));
        
        double dy = std::min(rect1(i,1)+rect1(i,3), rect2(i,1)+rect2(i,3))- std::max(rect1(i,1),rect2(i,1));
        
        inter_area(i,0) = dx*dy;
    }
    
    union_area = rect1.col(2)%rect1.col(3) + rect2.col(2)%rect2.col(3) - inter_area;
    
    return inter_area/union_area;
}

void mexFunction(int nlhs, mxArray *plhs[], int nrhs, const mxArray *prhs[]) {
    //getting beta arguments
    int M = mxGetM(prhs[0]);
    int N = mxGetN(prhs[0]);
    int X_dim[2] = {M,N}; 
    double *X = mxGetPr(prhs[0]);
    
    //getting beta arguments
    M = mxGetM(prhs[1]);
    N = mxGetN(prhs[1]);
    int bbox_dim[2] = {M,N}; 
    double *bbox = mxGetPr(prhs[1]);
    
    
    //getting beta arguments
    M = mxGetM(prhs[2]);
    N = mxGetN(prhs[2]);
    int bboxgt_dim[2] = {M,N}; 
    double *bbox_gt = mxGetPr(prhs[2]);
    
    //ACTUAL FUNCTION
    
    arma::mat X_2d = mat(X_dim[0], X_dim[1]);
    for(int i = 0; i<X_dim[0]; i++){
        for(int j = 0; j<X_dim[1]; j++){
            X_2d(i, j) = X[i+X_dim[0]*j];
        }
    }
    
    
    arma::mat bbox_2d = mat(bbox_dim[0], bbox_dim[1]);
    for(int i = 0; i<bbox_dim[0]; i++){
        for(int j = 0; j<bbox_dim[1]; j++){
            bbox_2d(i, j) = bbox[i+bbox_dim[0]*j];
        }
    }
    
    arma::mat bboxgt_2d = mat(bboxgt_dim[0], bboxgt_dim[1]);
    for(int i = 0; i<bboxgt_dim[0]; i++){
        for(int j = 0; j<bboxgt_dim[1]; j++){
            bboxgt_2d(i, j) = bbox_gt[i+bboxgt_dim[0]*j];
        }
    }
    
    double overlap = .6;
    double lambda = 1000;
    double robust = 0;
    
    //get_examples
    int n = bbox_2d.n_rows;
    
    arma::mat Y = arma::mat(n, 4);
    arma::mat O = overlap_ratio(bbox_2d, bboxgt_2d);

    arma::mat ex_box = arma::mat(1,4);
    arma::mat gt_box = arma::mat(1,4);
    
    for(int i = 0; i < n; i++){
        ex_box = 1.0*bbox_2d.row(i);
        gt_box = 1.0*bboxgt_2d.row(i);
        
        double src_w = ex_box(0,2);
        double src_h = ex_box(0,3);
        double src_ctr_x = ex_box(0,0) + .5*src_w;
        double src_ctr_y = ex_box(0,1) + .5*src_h;
        
        double gt_w = gt_box(0,2);
        double gt_h = gt_box(0,3);
        double gt_ctr_x = gt_box(0,0) + 0.5*gt_w;
        double gt_ctr_y = gt_box(0,1) + 0.5*gt_h;
        
        double dst_ctr_x = (gt_ctr_x - src_ctr_x) * 1.0/src_w;
        double dst_ctr_y = (gt_ctr_y - src_ctr_y) * 1.0/src_h;
        double dst_scl_w = log(gt_w / src_w);
        double dst_scl_h = log(gt_h / src_h);
        
        Y(i, 0) = dst_ctr_x;
        Y(i, 1) = dst_ctr_y;
        Y(i, 2) = dst_scl_w;
        Y(i, 3) = dst_scl_h;
    }
    
    arma::uvec idx = find(O < overlap);
    
    for(int i = idx.n_rows-1; i > -1; i--){
        Y.shed_row(idx(i));
        X_2d.shed_row(idx(i));
    }
        
    arma::mat mu = mean(Y);
    
    Y.each_row() -= mu;
    // END ACTUAL FUNCTION
    
  
    plhs[0] = mxCreateDoubleMatrix(Y.n_rows, Y.n_cols, mxREAL);
    double *output1 = mxGetPr(plhs[0]);
    for(int i = 0; i< Y.n_rows; i++){
        for(int j = 0; j<Y.n_cols; j++){
            output1[i+Y.n_rows*j] = Y(i,j);
        }
    }
    
    
    plhs[1] = mxCreateDoubleMatrix(mu.n_rows, mu.n_cols, mxREAL);
    double *output2 = mxGetPr(plhs[1]);
    for(int i = 0; i< mu.n_rows; i++){
        for(int j = 0; j<mu.n_cols; j++){
            output2[i+mu.n_rows*j] = mu(i,j);
        }
     }
    
     plhs[2] = mxCreateDoubleMatrix(X_2d.n_rows, X_2d.n_cols, mxREAL);
     double *output3 = mxGetPr(plhs[2]);
     for(int i = 0; i< X_2d.n_rows; i++){
         for(int j = 0; j<X_2d.n_cols; j++){
             output3[i+X_2d.n_rows*j] = X_2d(i,j);
         }
      }
    
}