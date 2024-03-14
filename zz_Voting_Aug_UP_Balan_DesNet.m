%% Test Augmentation

clear all
close all


%dataDir= './tr_processed/';

dataDir= './Data/';

imds = imageDatastore(dataDir, ...
    'IncludeSubfolders',true,'LabelSource','foldernames');

% [imdsTR, imdsTS] =  splitEachLabel(imds,.8);
% 
% idms_US=imdsTR;
% imds_ORG=imdsTS;
%%
%% augemntation
%% US
% labelCount = countEachLabel(imdsTR);
% figure;
% histogram(imdsTR.Labels);
% labels=imdsTR.Labels;
% [G,classes] = findgroups(labels);
% numObservations = splitapply(@numel,labels,G);
% 
% desiredNumObservationsPerClass = max(numObservations);
% 
% files = splitapply(@(x){randReplicateFiles(x,desiredNumObservationsPerClass)},imdsTR.Files,G);
% files = vertcat(files{:});
% labels=[];info=strfind(files,'\');
% for i=1:numel(files)
%     idx=info{i};
%     dirName=files{i};
%     targetStr=dirName(idx(end-1)+1:idx(end)-1);
%     targetStr2=cellstr(targetStr);
%     labels=[labels;categorical(targetStr2)];
% end
% idms_US.Files = files;
% idms_US.Labels=labels;
% labelCount_oversampled = countEachLabel(idms_US);

% figure
% histogram(idms_US.Labels);

pixelRange = [-30 30];
RotationRange = [-30 30];
scaleRange = [0.8 1.2];
imageAugmenter = imageDataAugmenter( ...
    'RandXReflection',true, ...
    'RandXTranslation',pixelRange, ...
    'RandYTranslation',pixelRange, ...
    'RandXScale',scaleRange, ...
    'RandYScale',scaleRange, ...
    'RandRotation',RotationRange ...
    ); 


net =densenet201; % densenet201; %resnet50; % resnet101  densenet201
inputSize = net.Layers(1).InputSize;

%analyzeNetwork(net);


% augimdsTrain = augmentedImageDatastore(inputSize(1:2),imdsTrain, ...
%      'DataAugmentation',imageAugmenter);
% augimdsValid = augmentedImageDatastore(inputSize(1:2),imdsValid);
% augimdsTest = augmentedImageDatastore(inputSize(1:2),imdsTest);
%%

%  idx_imds=1:5:8985;
%  imds_imds = subset(imds,idx_imds);
%  imds=imds_imds;
%resss= [640 480 1];  % to be increased 

  %%  
%imds_TR_res_US=augmentedImageDatastore(inputSize(1:2),idms_US,'DataAugmentation',imageDataAugmenter);

%imds_TS_res_ORG=augmentedImageDatastore(inputSize(1:2),imds_ORG,'DataAugmentation',imageDataAugmenter);

imds_aug=augmentedImageDatastore(inputSize(1:2),imds);


%imds_TR_res_US=idms_US
%imds_TS_res_ORG=imds_ORG

% Parameters
kfold= 5;
LR= 1e-3;  % -4

num_batch = 16;  % 16 %
Maxepochs= 5; 


%c = cvpartition(10,'KFold',3)
% label_TR=idms_US.Labels;
% label_TS=imds_ORG.Labels;
% label=[label_TR',label_TS']';
% fold_TR    = cvpartition(label_TR,'kfold',kfold);

%fold_TS    = cvpartition(label_TS,'kfold',kfold);
%fold =cvpartition(label_TS,'Holdout',50,'Stratify',true)

Afold   = zeros(kfold,1); 
confmat = 0
%%

label=imds.Labels;
fold = cvpartition(label,'kfold',5);


% hpartition = cvpartition(label,'Holdout',0.2); % Nonstratified partition
% idxTrain = training(hpartition);
% tblTrain = label(idxTrain,:);
% idxNew = test(hpartition);
% tblNew = label(idxNew,:);
Results0=0;
Prob_Net_Desnet201=[];
for i = 1:kfold
    train_idx  = fold.training(i);
  
    test_idx   = fold.test(i);
    
    xtrain     = subset(imds,train_idx);
    Train_US=xtrain;
    
    
    labelCount = countEachLabel(xtrain);
    figure;
    histogram(xtrain.Labels);
    labels=xtrain.Labels;
    [G,classes] = findgroups(labels);
    numObservations = splitapply(@numel,labels,G);

    desiredNumObservationsPerClass = max(numObservations);

    files = splitapply(@(x){randReplicateFiles(x,desiredNumObservationsPerClass)},Train_US.Files,G);
    files = vertcat(files{:});
    labels=[];info=strfind(files,'\');
    for i=1:numel(files)
        idx=info{i};
        dirName=files{i};
        targetStr=dirName(idx(end-1)+1:idx(end)-1);
        targetStr2=cellstr(targetStr);
        labels=[labels;categorical(targetStr2)];
    end
    Train_US.Files = files;
    Train_US.Labels=labels;
    labelCount_oversampled = countEachLabel(xtrain);
    
    Train_US_Aug=augmentedImageDatastore(inputSize(1:2),Train_US,'DataAugmentation',imageDataAugmenter);
    
    figure;
    histogram(Train_US.Labels);
    
    label_US=Train_US.Labels;
   % fold_US = cvpartition(label_US,'KFold',kfold,'Stratify',false);
   % train_idx_US  = fold_US.training(i);

    
   % Train_US_Aug_sub     = subset(Train_US_Aug,train_idx);

  
  ytrain     = label_US(:);
  
  xtest      = subset(imds_aug,test_idx);
 % xtest_aug=augmentedImageDatastore(inputSize(1:2),xtest,'DataAugmentation',imageDataAugmenter);

  ytest      = label(test_idx);
  
  ytest_val=ytrain;
  xtest_val=Train_US_Aug;
   
  ytrain     = categorical(ytrain);
  ytest      = categorical(ytest);
  
 

  % Net Architecture 
    lgraph = layerGraph(net);
    
    % Number of categories
    numClasses = numel(categories(Train_US.Labels))
    % New Learnable Layer
    newLearnableLayer = fullyConnectedLayer(numClasses, ...
        'Name','new_fc', ...
        'WeightLearnRateFactor',2, ...
        'BiasLearnRateFactor',2);
    
    % Replacing the last layers with new layers
    
    lgraph = replaceLayer(lgraph,'fc1000',newLearnableLayer);
    newsoftmaxLayer = softmaxLayer('Name','new_softmax');
    lgraph = replaceLayer(lgraph,'fc1000_softmax',newsoftmaxLayer);
    newClassLayer = classificationLayer('Name','new_classoutput');
    lgraph = replaceLayer(lgraph,'ClassificationLayer_fc1000',newClassLayer);
    
%%
  
  options = trainingOptions('sgdm',...   % sgdm adam
    'InitialLearnRate',LR,...
    'MaxEpochs',Maxepochs,...
    'Plots','training-progress',...
     'ExecutionEnvironment','gpu',...
     'ValidationData',{xtest_val,ytest_val},...
    'MiniBatchSize',num_batch);

 NET = trainNetwork(Train_US_Aug,lgraph,options);
  %Pred       = classify(NET,xtest);
  
     [Pred,POSTERIOR_prob]=classify(NET,xtest);
     Prob_Net_Desnet201=[Prob_Net_Desnet201;POSTERIOR_prob];
     save('Prob_Net_Desnet201','Prob_Net_Desnet201');
     
     fold_desNet=fold
     save('fold_desNet','fold_desNet');


 
%featureLayer = 'avg_pool' %'fc7'; pool5  fc1000 avg_pool
%trainingFeatures = activations(net, xtrain, featureLayer, 'MiniBatchSize', 1, 'OutputAs','rows');
%classifier = fitcecoc(trainingFeatures, ytrain);

%classifier = fitcecoc(trainingFeatures,ytrain,'OptimizeHyperparameters','auto',...
  %  'HyperparameterOptimizationOptions',struct('AcquisitionFunctionName',...
  %  'expected-improvement-plus'))


%testFeatures = activations(net, xtest, featureLayer, 'MiniBatchSize',1, 'OutputAs', 'rows');
%Pred = predict(classifier, testFeatures);

   
con        = confusionmat(ytest,Pred);

figure
confmat_ch = confusionchart(con);

confmat_ch.RowSummary = 'row-normalized';
confmat_ch.ColumnSummary = 'column-normalized';
  
  confmat    = confmat + con; 
  Afold(i,1) = sum(diag(con)) / sum(con(:));
 % confmat_ch = confusionchart(con);
 Results0=Confusion_Matrix_Calculations(con);
end


Acc  = mean(Afold)
%time = toc;

CNN.acc = Acc; 
CNN.con = confmat;

figure
confmat_ch = confusionchart(confmat);

%CNN.t   = time;

fprintf('\n Classification Acc (CNN): %g %% \n ',100* Acc);


% overall
accuray = CNN.acc;
% 
confmat = CNN.con;

Results=Confusion_Matrix_Calculations(confmat);
%save('Results','Results');
