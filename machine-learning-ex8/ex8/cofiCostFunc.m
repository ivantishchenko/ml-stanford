function [J, grad] = cofiCostFunc(params, Y, R, num_users, num_movies, ...
                                  num_features, lambda)
%COFICOSTFUNC Collaborative filtering cost function
%   [J, grad] = COFICOSTFUNC(params, Y, R, num_users, num_movies, ...
%   num_features, lambda) returns the cost and gradient for the
%   collaborative filtering problem.
%

% Unfold the U and W matrices from params
X = reshape(params(1:num_movies*num_features), num_movies, num_features);
Theta = reshape(params(num_movies*num_features+1:end), ...
                num_users, num_features);

            
% You need to return the following values correctly
J = 0;
X_grad = zeros(size(X));
Theta_grad = zeros(size(Theta));

% ====================== YOUR CODE HERE ======================
% Instructions: Compute the cost function and gradient for collaborative
%               filtering. Concretely, you should first implement the cost
%               function (without regularization) and make sure it is
%               matches our costs. After that, you should implement the 
%               gradient and use the checkCostFunction routine to check
%               that the gradient is correct. Finally, you should implement
%               regularization.
%
% Notes: X - num_movies  x num_features matrix of movie features
%        Theta - num_users  x num_features matrix of user features
%        Y - num_movies x num_users matrix of user ratings of movies
%        R - num_movies x num_users matrix, where R(i, j) = 1 if the 
%            i-th movie was rated by the j-th user
%
% You should set the following variables correctly:
%
%        X_grad - num_movies x num_features matrix, containing the 
%                 partial derivatives w.r.t. to each element of X
%        Theta_grad - num_users x num_features matrix, containing the 
%                     partial derivatives w.r.t. to each element of Theta
%
%M = X .* R(:, size(X, 2) - 1);
%K = Y .* R;  
filter = ((X * Theta' - Y) .* R);
J = sum(sum(filter .^ 2 )) / 2;

%X_grad = filter * Theta;
%Theta_grad = filter' * X;

for i = 1:num_movies
  idx = find(R(i, :) == 1);
  Theta_tmp = Theta(idx, :);
  Y_tmp = Y(i, idx);
  
  X_grad(i, :) = (X(i, :) * Theta_tmp' - Y_tmp) * Theta_tmp;
end

for j = 1:num_users
  % list of all movies rated by user j
  idx = find(R(:, j) == 1);
  X_tmp = X(idx, :);
  Y_tmp = Y(idx, j);

  Theta_grad(j, :) = (X_tmp * Theta(j, :)' - Y_tmp)' * X_tmp;
  
  %idx = find(R(:, j) == 1);
  %X_tmp = X_tmp(idx, :);
  %Y_tmp = Y(idx, j);
  %Theta_grad(j, :) = (X(:, j) * Theta' - Y_tmp) * X(:, j);
end

%idx = find(R(1:num_movies, :) == 1);
%heta_tmp = Theta(idx, :);
%Y_tmp = Y(1:num_movies, idx);
%X_grad(1:num_movies, :) = (X(1:num_movies, :) * Theta_tmp' - Y_tmp) * Theta_tmp;

% =============================================================

grad = [X_grad(:); Theta_grad(:)];

end
