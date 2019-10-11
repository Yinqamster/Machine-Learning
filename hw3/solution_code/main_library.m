% 注：本代码参考的是examples/mnist中三个文件的写法 %
function net = test(varargin)
run matconvnet-1.0-beta24/matlab/vl_setupnn;
% 初始化设置
opts.batchNormalization = false ;
opts.network = [] ;
opts.networkType = 'simplenn' ;
[opts, varargin] = vl_argparse(opts, varargin) ;

sfx = opts.networkType ;
if opts.batchNormalization, sfx = [sfx '-bnorm'] ; end
[opts, varargin] = vl_argparse(opts, varargin) ;

opts.train = struct() ;
opts = vl_argparse(opts, varargin) ;
if ~isfield(opts.train, 'gpus'), opts.train.gpus = []; end;

f=1;
% 训练部分 %
% 搭建网络 %
%卷积层实现全连接层：20*20是输入大小，1是输入通道数，512是输出通道数%
net.layers{1} = struct('type', 'conv', ...
                           'weights', {{f*randn(20,20,1,512, 'single'), zeros(1, 512, 'single')}}, ...
                           'stride', 1, ...
                           'pad', 0) ;
% RELU层 %
net.layers{2} = struct(...
    'name', 'relu1', ...
    'type', 'relu') ;

net.layers{3} = struct('type', 'conv', ...
                           'weights', {{f*randn(1,1,512,512, 'single'), zeros(1, 512, 'single')}}, ...
                           'stride', 1, ...
                           'pad', 0) ;
net.layers{4} = struct(...
    'name', 'relu1', ...
    'type', 'relu') ;

net.layers{5} = struct('type', 'conv', ...
                           'weights', {{f*randn(1,1,512,10, 'single'), zeros(1, 10, 'single')}}, ...
                           'stride', 1, ...
                           'pad', 0) ;
                  
net.layers{6} = struct('type', 'softmaxloss') ;

%将刚才未设置的其他域用默认值填充，如果没有这一步可能无法运行%
net = vl_simplenn_tidy(net) ;

% 准备数据 %
train_data = single(csvread('train_data.csv'));
[train_num,dim]=size(train_data);
train_data = reshape(train_data',20,20,1,train_num);
% 训练函数规定label必须从1开始，因此我们为每个label加1 %
train_target = single(csvread('train_targets.csv')')+1;
% imdb.set用来区分验证集和训练集，这里我们设置训练集的标记为1，验证集的标记为3 %
set = [ones(1,train_num*3/4) 3*ones(1,train_num*1/4)];

imdb.images.data = train_data ;
imdb.images.labels = train_target ;
imdb.images.set = set;

% 训练过程 %
 [net, info] = cnn_train(net, imdb, getBatch(opts), ...
   opts.train, ...
   'val', find(imdb.images.set == 3),...
   'numEpochs',10) ;
 % 测试过程 %
test_data = csvread('test_data.csv');
net.layers{end}.type = 'softmax';
[test_num,dim] = size(test_data);
for i=1:test_num
    xx=single(reshape(test_data(i,:),20,20));
    res = vl_simplenn(net, xx);
    [score,index] = max(res(end).x);
% 由于训练函数的label从1开始，因此这里减1 %
    label(i)=index-1;
end
csvwrite('test_predictions_library.csv',label');
end

% 两个辅助函数，来源于mnist example %
% --------------------------------------------------------------------
function fn = getBatch(opts)
% --------------------------------------------------------------------
    fn = @(x,y) getSimpleNNBatch(x,y) ;
end

function [images, labels] = getSimpleNNBatch(imdb, batch)
% --------------------------------------------------------------------
images = imdb.images.data(:,:,:,batch) ;
labels = imdb.images.labels(1,batch) ;
end