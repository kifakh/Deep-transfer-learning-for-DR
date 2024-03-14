% 26 3 2021
% Retinopathy
% test SVM with deep network 



clear, close all
%% 
%dataDir= './tr_processed/';
dataDir= './Data/';
imds = imageDatastore(dataDir, ...
    'IncludeSubfolders',true,'LabelSource','foldernames');

% layers = [
%     imageInputLayer(resss); % Input to the network is a 256x256x1 sized image 
%     convolution2dLayer(5,16,'Padding',[2 2],'Stride', [2,2]);  % convolution layer with 20, 5x5 filters
%     batchNormalizationLayer;  % 16 is previous, we try 64
%     reluLayer();  % ReLU layer
%     convolution2dLayer(5,32,'Padding',[2 2],'Stride', [2,2]);  % convolution layer with 20, 5x5 filters
%     batchNormalizationLayer;
%     reluLayer();
%     maxPooling2dLayer(2,'Stride',2); % Max pooling layer
%     convolution2dLayer(3,64,'Padding',[1 1],'Stride', [1,1]);  % convolution layer with 20, 5x5 filters
%     reluLayer();
%     maxPooling2dLayer(2,'Stride',2); 
%     convolution2dLayer(3,64,'Padding',[1 1],'Stride', [1,1]);  % convolution layer with 20, 5x5 filters
%     reluLayer();
%     maxPooling2dLayer(2,'Stride',1); 
%     
%     fullyConnectedLayer(200);% it was 200 % Fullly connected layer with 50 activations
%     dropoutLayer(.1); % Dropout layer
%     fullyConnectedLayer(5); % Fully connected with 17 layers
%     softmaxLayer(); % Softmax normalization layer
%     classificationLayer(); % Classification layer
%     ];

convnet = resnet50; % alexnet;   
%analyzeNetwork(convnet)

imageSize = convnet.Layers(1).InputSize;
%% 
imds.ReadFcn = @(filename)readAndPreprocessImage(filename, imageSize);

%% 
[trainingSet_pre, testSet_pre] = splitEachLabel(imds, 0.8, 'randomize');
% features

pixelRange = [-20 20];
scaleRange = [0.8 1.2];
% imageAugmenter = imageDataAugmenter( ...
%     'RandXReflection',true, ...
%     'RandXTranslation',pixelRange, ...
%     'RandYTranslation',pixelRange, ...
%     'RandXScale',scaleRange, ...
%     'RandYScale',scaleRange);  

    imageAugmenter = imageDataAugmenter(...
        'FillValue',128,...
        'RandXReflection',true, ...
        'RandYReflection',true, ...
        'RandRotation',[-45 45],...
        'RandXTranslation',[-20 20], ...
        'RandYTranslation',[-20 20], ...
        'RandXShear',[0 20],...
        'RandYShear',[0 20],...
        'RandXScale',[0.9 1.1], ...
        'RandYScale',[0.9 1.1]);

  %%  
trainingSet=augmentedImageDatastore(imageSize(1:2),trainingSet_pre,'DataAugmentation',imageAugmenter);
testSet=augmentedImageDatastore(imageSize(1:2),testSet_pre,'DataAugmentation',imageAugmenter);



featureLayer = 'fc1000'; %'fc7';
trainingFeatures = activations(convnet, trainingSet, featureLayer, 'MiniBatchSize', 1, 'OutputAs','rows');
%  classify
%
% 
%  options = trainingOptions('sgdm',...   % sgdm
%     'InitialLearnRate',LR,...
%     'MaxEpochs',Maxepochs,...
%     'Plots','training-progress',...
%      'ExecutionEnvironment','gpu',...
%      'ValidationData',{xtest_val,ytest_val},...
%     'MiniBatchSize',num_batch);

classifier = fitcecoc(trainingFeatures, trainingSet_pre.Labels);
%  
% acc
testFeatures = activations(convnet, testSet, featureLayer, 'MiniBatchSize',1, 'OutputAs', 'rows');
predictedLabels = predict(classifier, testFeatures);

accuracy = mean(predictedLabels == testSet_pre.Labels)

% Confusion
C = confusionmat(testSet_pre.Labels, predictedLabels);
Afold = sum(diag(C)) / sum(C(:))

figure
confmat_ch = confusionchart(C);
%%
Results=Confusion_Matrix_Calculations(C)