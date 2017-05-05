function bbox_reg = train_bbox_reg_shell(X, bbox, bbox_gt, varargin)

ip = inputParser;
ip.addParamValue('min_overlap', 0.6,   @isscalar);
ip.addParamValue('lambda',      1000,  @isscalar);
ip.addParamValue('robust',      0,     @isscalar);

ip.parse(varargin{:});
opts = ip.Results;

[Y, mu, X] = train_bbox_reg(double(X), bbox, bbox_gt);
X = cat(2,X,ones(size(X,1),1,class(X)));

S = Y'*Y / size(Y,1);
[V, D] = eig(S);
D = diag(D);
T = V*diag(1./sqrt(D+.001))*V';
T_inv = V*diag(sqrt(D+0.001))*V';
Y = Y * T;
model.mu = mu;
model.T = T;
model.T_inv = T_inv;


model.Beta = [ ...
    solve_robust(X, Y(:,1), opts.lambda, 'ridge_reg_chol', opts.robust) ...
    solve_robust(X, Y(:,2), opts.lambda, 'ridge_reg_chol', opts.robust) ...
    solve_robust(X, Y(:,3), opts.lambda, 'ridge_reg_chol', opts.robust) ...
    solve_robust(X, Y(:,4), opts.lambda, 'ridge_reg_chol', opts.robust)];
    

bbox_reg.model = model;
bbox_reg.training_opts = opts;

end

% ------------------------------------------------------------------------
function [x, losses] = solve_robust(A, y, lambda, method, qtile)
% ------------------------------------------------------------------------
[x, losses] = solve(A, y, lambda, method);
% fprintf('loss = %.3f\n', mean(losses));
if qtile > 0
  thresh = quantile(losses, 1-qtile);
  I = find(losses < thresh);
  [x, losses] = solve(A(I,:), y(I), lambda, method);
  fprintf('loss (robust) = %.3f\n', mean(losses));
end
end

% ------------------------------------------------------------------------
function [x, losses] = solve(A, y, lambda, method)
% ------------------------------------------------------------------------

%tic;_
switch method
case 'ridge_reg_chol'
  % solve for x in min_x ||Ax - y||^2 + lambda*||x||^2
  %
  % solve (A'A + lambdaI)x = A'y for x using cholesky factorization
  % R'R = (A'A + lambdaI)
  % R'z = A'y  :  solve for z  =>  R'Rx = R'z  =>  Rx = z
  % Rx = z     :  solve for x
  R = chol(A'*A + lambda*eye(size(A,2)));
  z = R' \ (A'*y);
  x = R \ z;
case 'ridge_reg_inv'
  % solve for x in min_x ||Ax - y||^2 + lambda*||x||^2
  x = inv(A'*A + lambda*eye(size(A,2)))*A'*y;
case 'ls_mldivide'
  % solve for x in min_x ||Ax - y||^2
  if lambda > 0
    warning('ignoring lambda; no regularization used');
  end
  x = A\y;
end
%toc;
losses = 0.5 * (A*x - y).^2;
end
