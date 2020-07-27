function [convnetChannelEstimation] = train_ChannelNet(X,Y,l_rate)
%% Train network
dataFRFChainSelection = X;
labelsRFChainSelection = Y;
sizeInputFRFChainSelection = size(dataFRFChainSelection);
sizeOutputFRFChainSelection = size(labelsRFChainSelection);
% val. for regression.
idx = randperm(size(dataFRFChainSelection,4),floor(.2*sizeInputFRFChainSelection(end)));
valDataRFChainSelection = dataFRFChainSelection(:,:,:,idx);
valLabelsFRFChainSelection = labelsRFChainSelection(idx,:);
dataFRFChainSelection(:,:,:,idx) = [];
labelsRFChainSelection(idx,:) = [];
% settings.
miniBatchSize = 16;
numValidationsPerEpoch = 5000;
validationFrequency = 50*1;
%% DNN for HB.
layersFRFChainSelection = [imageInputLayer([sizeInputFRFChainSelection(1:3)],'Normalization', 'zerocenter');
    convolution2dLayer(3,2^8,'Padding','same');
%     batchNormalizationLayer
%     reluLayer();
%     maxPooling2dLayer([2 2],'Stride',2);
    convolution2dLayer(3,2^8,'Padding','same');
%     batchNormalizationLayer
%     reluLayer();
% maxPooling2dLayer([2 2],'Stride',2);
%     convolution2dLayer(2^2,2^7);
%     batchNormalizationLayer
%     reluLayer();
% maxPooling2dLayer([2 2],'Stride',2);
    fullyConnectedLayer(2^10);
%     fullyConnectedLayer(2^14);
%     fullyConnectedLayer(2^10);
% maxPooling2dLayer([2 2],'Stride',2);

    %     reluLayer();
%     dropoutLayer();
%     fullyConnectedLayer(2^7);
%     dropoutLayer();
%     fullyConnectedLayer(2^10);
    fullyConnectedLayer(2^10);
%     reluLayer();
%     dropoutLayer();
%     fullyConnectedLayer(QFRF);
%     softmaxLayer();
%     classificationLayer();
    fullyConnectedLayer(sizeOutputFRFChainSelection(2),'Name','fc_2')
    regressionLayer('Name','reg_out')
    ];
optsFRFSelection = trainingOptions('sgdm',...
    'Momentum', 0.9,...
    'InitialLearnRate',l_rate,... % The default value is 0.01.
    'MaxEpochs',5000,...
    'MiniBatchSize',miniBatchSize,... % The default is 128.
    'LearnRateSchedule','piecewise',...
    'LearnRateDropFactor',.5,...
    'LearnRateDropPeriod',10,...
    'L2Regularization',0.000000001,... % The default is 0.0001.
    'ExecutionEnvironment', 'auto',...
    'ValidationData',{valDataRFChainSelection,valLabelsFRFChainSelection},...
    'ValidationFrequency',validationFrequency,...
    'ValidationPatience', 20000,...
    'Plots','none',...
    'Shuffle','every-epoch',...
    'OutputFcn',@(info)stopIfAccuracyNotImproving(info,3));
%%
% fprintf(2,['Train CENet for Channel Estimation \n'])
timeCENET = tic;
convnetChannelEstimation = trainNetwork(dataFRFChainSelection, labelsRFChainSelection, layersFRFChainSelection, optsFRFSelection);
timeCENET = toc(timeCENET);