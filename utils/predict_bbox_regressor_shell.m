function pred_boxes = predict_bbox_regressor_shell(model, feat, ex_boxes)

beta = double(model.Beta);
t_inv = double(model.T_inv);
mu = double(model.mu);

feat = double(feat);
pred_boxes = predict_bbox_reg(beta, t_inv, mu, feat, double(ex_boxes)...
    , feat*beta(1:end-1, :));


end