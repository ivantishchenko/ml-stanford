function plotData(X, y)
%PLOTDATA Plots the data points X and y into a new figure 
%   PLOTDATA(x,y) plots the data points with + for the positive examples
%   and o for the negative examples. X is assumed to be a Mx2 matrix.

% Create New Figure
figure; hold on;

% ====================== YOUR CODE HERE ======================
% Instructions: Plot the positive and negative examples on a
%               2D plot, using the option 'k+' for the positive
%               examples and 'ko' for the negative examples.
%



neg = X(find(y == 0), :); % all zeros classified training examples
pos = X(find(y == 1), :); % all ones classified training examples

scatter(neg(:,1), neg(:,2), "r", "filled");
scatter(pos(:,1), pos(:,2), "g", "filled");

%names = {'Negative', 'Positive'};
%legend(names);

% =========================================================================



hold off;

end
