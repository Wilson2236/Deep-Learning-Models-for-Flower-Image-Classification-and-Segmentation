%% Define Network
inputSize = [256 256 3];
imgLayer = imageInputLayer(inputSize);

filterSize = 3;
numFilters = 32;
conv = convolution2dLayer(filterSize, numFilters, 'Padding', 1);
relu = reluLayer();

poolSize = 2;
maxPoolDownsample2x = maxPooling2dLayer(poolSize, 'Stride', 2);

downsamplingLayers = [
    conv
    relu
    maxPoolDownsample2x
    conv
    relu
    maxPoolDownsample2x
    ];

filterSize = 4;
transposedConvUpsample2x = transposedConv2dLayer(4, numFilters, 'Stride', 2, 'Cropping', 1);

upsamplingLayers = [
    transposedConvUpsample2x
    relu
    transposedConvUpsample2x
    relu
    ];

numClasses = 2;
conv1x1 = convolution2dLayer(1, numClasses);

finalLayers = [
    conv1x1
    softmaxLayer()
    pixelClassificationLayer()
    ];

net = [
    imgLayer
    downsamplingLayers
    upsamplingLayers
    finalLayers
    ];

dataSetDir = fullfile(toolboxdir('vision'), 'visiondata', 'triangleImages');
imageDir = fullfile('daffodilSeg', 'ImagesRsz256');
labelDir = fullfile("daffodilSeg", 'LabelsRsz256');

imds = imageDatastore(imageDir);
classNames = ["flower", "background"];
labelIDs = [1 3];
pxds = pixelLabelDatastore(labelDir, classNames, labelIDs);


%% Split the Dataset
trainingFraction = 0.8;
validationFraction = 0.1;
testingFraction = 0.1;

numImages = numel(imds.Files);
numTrainingImages = round(trainingFraction * numImages);
numValidationImages = round(validationFraction * numImages);
numTestingImages = numImages - numTrainingImages - numValidationImages;

% Create a random permutation of image indices for partitioning
randomIndices = randperm(numImages);

% Calculate the indices for each set
trainingIndices = randomIndices(1:numTrainingImages);
validationIndices = randomIndices(numTrainingImages+1:numTrainingImages+numValidationImages);
testingIndices = randomIndices(numTrainingImages+numValidationImages+1:end);

% Split the image datastore
imdsTrain = subset(imds, trainingIndices);
imdsValidation = subset(imds, validationIndices);
imdsTest = subset(imds, testingIndices);

% Split the pixel label datastore
pxdsTrain = subset(pxds, trainingIndices);
pxdsValidation = subset(pxds, validationIndices);
pxdsTest = subset(pxds, testingIndices);




%% Data Augmentation
augmenter = imageDataAugmenter( ...
    'RandXReflection', true, ...
    'RandYReflection', true, ...
    'RandRotation', [-10, 10], ...
    'RandXScale', [0.5 2], ...
    'RandYScale', [0.5 2]);

% Combine image and pixel label datastores for training and validation
trainingData = pixelLabelImageDatastore(imdsTrain, pxdsTrain, 'DataAugmentation', augmenter);
validationData = pixelLabelImageDatastore(imdsValidation, pxdsValidation);


%% Train Network
opts = trainingOptions('sgdm', ...
    'InitialLearnRate', 0.0001, ...
    'MaxEpochs', 200, ...
    'MiniBatchSize', 32, ...
    'ValidationData', validationData, ...
    'ValidationFrequency', 10, ...
    'Plots', 'training-progress', ...
    'L2Regularization', 1e-4, ...
    'Shuffle', 'every-epoch');

net = trainNetwork(trainingData, net, opts);

save('segmentnet.mat', 'net');

%% Show One Image
testImage = imread('image_0032.jpg');
imshow(testImage);
testImage = imresize(testImage, [256, 256]); % resize the image to match the label matrix size
C = semanticseg(testImage, net);
B = labeloverlay(testImage, C);
imshow(B);

%% Evaluate Test Set Performance
net = load('myModel.mat');
net = net.net;

pxdsResults = semanticseg(imdsTest,net,"WriteLocation",tempdir);
metrics = evaluateSemanticSegmentation(pxdsResults,pxdsTest);
metrics.ConfusionMatrix
metrics.ClassMetrics

% Calculation of precision, recall, F1 score, and Dice coefficient
numClasses = numel(classNames);
tp = diag(metrics.ConfusionMatrix.Variables); % true positives
fp = sum(metrics.ConfusionMatrix.Variables, 2) - tp; % false positives
fn = sum(metrics.ConfusionMatrix.Variables, 1)' - tp; % false negatives
precision = tp ./ (tp + fp);
recall = tp ./ (tp + fn);
F1 = 2 * (precision .* recall) ./ (precision + recall);
Dice = 2 * tp ./ (2 * tp + fp + fn);

% Display class-wise metrics
fprintf('Precision per class: %s\n', mat2str(precision, 2))
fprintf('Recall per class: %s\n', mat2str(recall, 2))
fprintf('F1 Score per class: %s\n', mat2str(F1, 2))
fprintf('Dice Coefficient per class: %s\n', mat2str(Dice, 2))

% Display mean metrics
fprintf('Mean Precision: %f\n', mean(precision))
fprintf('Mean Recall: %f\n', mean(recall))
fprintf('Mean F1 Score: %f\n', mean(F1))
fprintf('Mean Dice Coefficient: %f\n', mean(Dice))

cm = confusionchart(metrics.ConfusionMatrix.Variables, ...
  classNames, 'Normalization','row-normalized');
cm.Title = 'Normalized Confusion Matrix (%)';

imageIoU = metrics.ImageMetrics.MeanIoU;
figure
histogram(imageIoU)
title('Image Mean IoU')
