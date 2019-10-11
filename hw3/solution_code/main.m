% clear; clc;
% =============================================================
% initialize
% =============================================================
data = csvread('train_data.csv');
target = csvread('train_targets.csv');

% one-hot encoding
% 1000000000 = 0
% 0100000000 = 1
% ...
% 0000000001 = 9
targets = bsxfun(@eq, target(:), 0 : 9);

% activation function
f = @(z) (1.0 ./ (1.0 + exp(-z)));
% learning rate
eta = 0.1;

[row_d, col_d] = size(data);

% hidder layer bias
gamma = zeros(1, 100);
% output layer bias
theta = zeros(1, 10);
% connection weight between input layer and hidden layer
v = rand(400, 100) * 0.1 - 0.05;
% connection weight between hidden layer and output layer
w = rand(100, 10) * 0.1 - 0.05;

% =============================================================
% train
% =============================================================
for t = 1 : 30
    prediction = zeros(row_d, 10);
    for i = 1 : row_d
        x = data(i, :);
        y = targets(i, :);
        alpha = x * v;
        b = f(alpha - gamma);
        beta = b * w;
        yt = f(beta - theta);
        
        prediction(i, :) = yt;
        
        % compute gj and eh
        g = yt .* (1 - yt) .* (y - yt); %5.10
        e = b .* (1 - b) .* (g * w');  %5.15
        
        % update w
        delta_w = eta * (b' * g);
        w = w + delta_w;
        
        % update theta
        delta_theta = -eta * g;
        theta = theta + delta_theta;
        
        % update v
        delta_v = eta * (x' * e);
        v = v + delta_v;
        
        % update gamma
        delta_gamma = -eta * e;
        gamma = gamma + delta_gamma;
    end
    % [~, i] = max(prediction, [], 2);
    % prediction = i - 1;
    % acc = sum(target == prediction) / size(target, 1)
end

[~, i] = max(prediction, [], 2);
prediction = i - 1;
acc = sum(target == prediction) / size(target, 1);
% csvwrite(['train_predictions.csv'], prediction);
fprintf('train accuracy: %f\n',acc);

% =============================================================
% test
% =============================================================
test_data = csvread('test_data.csv');
row_t = size(test_data, 1);
prediction = zeros(row_t, 10);
for i = 1 : row_t
    x = test_data(i, :);
    alpha = x * v;
    b = f(alpha - gamma);
    beta = b * w;
    yt = f(beta - theta);
    prediction(i, :) = yt;
end
[~, i] = max(prediction, [], 2);
prediction = i - 1;
csvwrite(['test_predictions.csv'], prediction);
% colormap(gray);
% imagesc(reshape(data(4, :), 20, 20));