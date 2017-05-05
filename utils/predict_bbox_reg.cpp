#include <mex.h>
#include <iostream>
#include <armadillo>

using namespace arma;

arma::mat predict_bbox_reg(double *beta, int beta_dim[2], double *t_inv, int tinv_dim[2],
            double *mu, double *feat, int feat_dim[2], double *boxes, int box_dim[2],
        double *Y, int Y_dim[2]){
    
    
    if(box_dim[0] == 0 || box_dim[1] == 0){
        arma::mat bad = mat(1,1);
        return bad;
    }
    
    //beta
    arma::mat beta_2d = mat(beta_dim[0], beta_dim[1]);
    for(int i = 0; i<beta_dim[0]; i++){
        for(int j = 0; j<beta_dim[1]; j++){
            beta_2d(i, j) = beta[i+beta_dim[0]*j];
        }
    }
    
    //t_inv
    arma::mat tinv_2d = mat(tinv_dim[0], tinv_dim[1]);
    for(int i = 0; i<tinv_dim[0]; i++){
        for(int j = 0; j<tinv_dim[1]; j++){
            tinv_2d(i, j) = t_inv[i+tinv_dim[0]*j];
        }
    }
    
    //mu
    arma::mat mu_mat = mat(1, 4);
    for(int i = 0; i<4; i++){
            mu_mat(0, i) = *(mu+i);
    }
        
  
    
    //feat_2d
    arma::mat feat_2d = mat(feat_dim[0], feat_dim[1]);
    for(int i = 0; i<feat_dim[0]; i++){
        for(int j = 0; j<feat_dim[1]; j++){
            feat_2d(i, j) = feat[i+feat_dim[0]*j];
        }
    }
    
    //boxes
    arma::mat box_2d = mat(box_dim[0], box_dim[1]);
    for(int i = 0; i<box_dim[0]; i++){
        for(int j = 0; j<box_dim[1]; j++){
            box_2d(i, j) = boxes[i+box_dim[0]*j];
        }
    }
    
    //Y
    arma::mat Y_2d = mat(Y_dim[0], Y_dim[1]);
    for(int i = 0; i<Y_dim[0]; i++){
        for(int j = 0; j<Y_dim[1]; j++){
            Y_2d(i, j) = Y[i+Y_dim[0]*j];
        }
    }
    
    //std::cout << "last beta row\n";
    //beta_2d.row(beta_2d.n_rows-1).print();
    //std::cout << "mult\n";
    //Y_2d.print();
    //std::cout << "final result\n";
    Y_2d.each_row() += beta_2d.row(beta_2d.n_rows-1);
    //Y_2d.print();
    
    //std::cout << "mu\n";
    //mu_mat.print();
    
    //std::cout << "second bsxfun\n";
    Y_2d = Y_2d*tinv_2d;
    Y_2d.each_row() += mu_mat.row(0);
   // Y_2d.print();
   
    arma::mat dst_ctr_x = Y_2d.col(0);
    arma::mat dst_ctr_y = Y_2d.col(1);
    arma::mat dst_scl_x = Y_2d.col(2);
    arma::mat dst_scl_y = Y_2d.col(3);
    
    arma::mat src_w = box_2d.col(2);
    arma::mat src_h = box_2d.col(3);
    arma::mat src_ctr_x = box_2d.col(0) + (.5*src_w);
    arma::mat src_ctr_y = box_2d.col(1) + (.5*src_h);
    
    arma::mat pred_ctr_x = (dst_ctr_x % src_w) + src_ctr_x;
    arma::mat pred_ctr_y = (dst_ctr_y % src_h) + src_ctr_y;
    arma::mat pred_w = exp(dst_scl_x) % src_w;
    arma::mat pred_h = exp(dst_scl_y) % src_h;
    
    arma::mat predict_boxes = arma::mat(5,4);
    predict_boxes.col(0) = pred_ctr_x - 0.5*pred_w;
    predict_boxes.col(1) = pred_ctr_y - 0.5*pred_h;
    predict_boxes.col(2) = pred_w;
    predict_boxes.col(3) = pred_h;
    
    return predict_boxes;
}
        
           
        
        
void mexFunction(int nlhs, mxArray *plhs[], int nrhs, const mxArray *prhs[]) {
    //getting beta arguments
    int M = mxGetM(prhs[0]);
    int N = mxGetN(prhs[0]);
    int beta_dim[2] = {M,N}; 
    double *beta = mxGetPr(prhs[0]);

    //getting t_inv arguments
    M = mxGetM(prhs[1]);
    N = mxGetN(prhs[1]);
    int tinv_dim[2] = {M,N};
    double *t_inv = mxGetPr(prhs[1]);
   
    //getting mu arguments
    double *mu = mxGetPr(prhs[2]);
    
    //getting feat arguments
    M = mxGetM(prhs[3]);
    N = mxGetN(prhs[3]);
    int feat_dim[2] = {M,N};
    double *feat = mxGetPr(prhs[3]);
    
    //getting ex_boxes arguments
    M = mxGetM(prhs[4]);
    N = mxGetN(prhs[4]);
    int box_dim[2] = {M,N};
    double *ex_boxes = mxGetPr(prhs[4]);
    
    //getting Y arguments
    M = mxGetM(prhs[5]);
    N = mxGetN(prhs[5]);
    int Y_dim[2] = {M,N};
    double *Y = mxGetPr(prhs[5]);
    
    //int *pred_boxes = 
    arma::mat predict_boxes = predict_bbox_reg(beta, beta_dim, t_inv, tinv_dim,
            mu, feat, feat_dim, ex_boxes, box_dim, Y, Y_dim);
    //predict_boxes.print();
    
    plhs[0] = mxCreateDoubleMatrix( predict_boxes.n_rows, predict_boxes.n_cols, mxREAL);
    double *output = mxGetPr(plhs[0]);
    for(int i = 0; i< predict_boxes.n_rows; i++){
        for(int j = 0; j<predict_boxes.n_cols; j++){
            output[i+predict_boxes.n_rows*j] = predict_boxes(i,j);
        }
    }
    
}
  
  
