% 26 3 2021
% Retinopathy
% test CNN
% K fold

clear all
close all

dataDir= './tr_processed/';

imds = imageDatastore(dataDir, ...
    'IncludeSubfolders',true,'LabelSource','foldernames');

idx_imds=1:10:3648
imds_imds = subset(imds,idx_imds);

resss= [400 250 1];  % to be increased 
pixelRange = [-30 30];
scaleRange = [0.9 1.1];
imageAugmenter = imageDataAugmenter( ...
    'RandXReflection',true, ...
    'RandXTranslation',pixelRange, ...
    'RandYTranslation',pixelRange, ...
    'RandXScale',scaleRange, ...
    'RandYScale',scaleRange);     
  %%  
imds_imds_res=augmentedImageDatastore(resss,imds_imds,'DataAugmentation',imageAugmenter);

% 
% X=readall(imds_imds);
% XTrain = reshape( cat(3,X{:})  , [614,819,3,3410]);
% YTrain=imds.Labels;
% dsXTrain = arrayDatastore(XTrain,'IterationDimension',4);
% dsYTrain = arrayDatastore(YTrain);
% dsTrain = combine(dsXTrain,dsYTrain);

imagesCellArray = imds_imds.readall();
numImages = numel(imagesCellArray );

[h, w, c] = size(imagesCellArray{1,1})
%X = zeros( h, w, c, numImages );
X = zeros( 400,250 , 1, numImages );

for i=1:numImages
A=imagesCellArray{i};
B = imresize(A,[400 250]);
X(:,:,:,i) = im2double(B);
end

T = imds_imds.Labels;

imgs=X;
label=T;

% Parameters
kfold= 5;
LR= 1e-4; 

num_batch = 32;  % 16 %
Maxepochs= 10; 

%CNN
tic;
resss= [400 250 1];  % to be increased 

height  = size(imgs,1);
width   = size(imgs,2); 
channel = size(imgs,3);


layers = [
    imageInputLayer(resss); % Input to the network is a 256x256x1 sized image 
    convolution2dLayer(5,16,'Padding',[2 2],'Stride', [2,2]);  % convolution layer with 20, 5x5 filters
    batchNormalizationLayer;  % 16 is previous, we try 64
    reluLayer();  % ReLU layer
    convolution2dLayer(5,32,'Padding',[2 2],'Stride', [2,2]);  % convolution layer with 20, 5x5 filters
    batchNormalizationLayer;
    reluLayer();
    maxPooling2dLayer(2,'Stride',2); % Max pooling layer
    convolution2dLayer(3,64,'Padding',[1 1],'Stride', [1,1]);  % convolution layer with 20, 5x5 filters
    reluLayer();
    maxPooling2dLayer(2,'Stride',2); 
    convolution2dLayer(3,64,'Padding',[1 1],'Stride', [1,1]);  % convolution layer with 20, 5x5 filters
    reluLayer();
    maxPooling2dLayer(2,'Stride',1); 
    
    fullyConnectedLayer(200);% it was 200 % Fullly connected layer with 50 activations
    dropoutLayer(.1); % Dropout layer
    fullyConnectedLayer(5); % Fully connected with 17 layers
    softmaxLayer(); % Softmax normalization layer
    classificationLayer(); % Classification layer
    ];


fold    = cvpartition(label,'kfold',kfold);
Afold   = zeros(kfold,1); 
confmat = 0;
for i = 1:kfold
  train_idx  = fold.training(i);
  test_idx   = fold.test(i);
  
  xtrain     = imgs(:,:,1,train_idx);
  ytrain     = label(train_idx);
  
  xtest      = imgs(:,:,1,test_idx); 
  ytest      = label(test_idx);
  
  ytest_val=ytest;
  xtest_val=xtest;
   
  ytrain     = categorical(ytrain);
  ytest      = categorical(ytest);
  
 
  options = trainingOptions('sgdm',...   % sgdm
    'InitialLearnRate',LR,...
    'MaxEpochs',Maxepochs,...
    'Plots','training-progress',...
     'ExecutionEnvironment','gpu',...
     'ValidationData',{xtest_val,ytest_val},...
    'MiniBatchSize',num_batch);
  
  net        = trainNetwork(xtrain,ytrain,layers,options);
  Pred       = classify(net,xtest);
  con        = confusionmat(ytest,Pred);
  
  figure
  confmat_ch = confusionchart(con);
  
  confmat    = confmat + con; 
  Afold(i,1) = sum(diag(con)) / sum(con(:));
end
Acc  = mean(Afold);
time = toc;

CNN.acc = Acc; 
CNN.con = confmat;

figure
confmat_ch = confusionchart(confmat);

CNN.t   = time;

fprintf('\n Classification Acc (CNN): %g %% \n ',100* Acc);



% overall
accuray = CNN.acc;
% 
confmat = CNN.con;
