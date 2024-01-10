%% Create FlowerDataset
clc
clear

% Define the parent folder name
parentFolderName = 'FlowerDataset';

% Create the parent folder if it doesn't already exist
if ~exist(parentFolderName, 'dir')
    mkdir(parentFolderName);
end

% Define the subfolder names
subfolderNames = {'flower1', 'flower2', 'flower3', 'flower4', 'flower5', ...
                  'flower6', 'flower7', 'flower8', 'flower9', 'flower10', ...
                  'flower11', 'flower12', 'flower13', 'flower14', 'flower15', ...
                  'flower16', 'flower17'};

% Create the subfolders if they don't already exist, and copy the files to them
for i = 1:length(subfolderNames)
    subfolderPath = fullfile(parentFolderName, subfolderNames{i});
    if ~exist(subfolderPath, 'dir')
        mkdir(subfolderPath);
    end
    
    startIndex = (i-1)*80 + 1; % Calculate the start index for this folder
    endIndex = startIndex + 79; % Calculate the end index for this folder
    
    % Loop through the images and copy them to the subfolder
    for j = startIndex:endIndex
        sourceFileName = sprintf('17flowers/image_%04d.jpg', j);
        destinationFolder = subfolderPath;
        copyfile(sourceFileName, destinationFolder);
    end
end

%% Training, Testing and Validation Set

% Load the dataset as an imageDatastore
imds = imageDatastore(parentFolderName, 'IncludeSubfolders', true, 'LabelSource', 'foldernames', 'ReadFcn', @(x) customResize(x));

% Set the ratios for training, validation, and testing sets
trainRatio = 0.7;
%valTestRatio = 0.15;

% Split the data into training, validation, and testing sets
[imdsTrain, imdsTemp] = splitEachLabel(imds, trainRatio, 'randomized');
[imdsVal, imdsTest] = splitEachLabel(imdsTemp, 0.5, 'randomized');


%% Pre-processing
% Data augmentation options
augmentationOptions = imageDataAugmenter( ...
    'RandXScale', [0.5 2], ...
    'RandYScale', [0.5 2], ...  
    'RandXReflection', true, ...
    'RandYReflection', true, ...
    'RandRotation', [-10, 10] ...
);

% Create augmented imageDatastores for training and validation
augImdsTrain = augmentedImageDatastore([256 256 3], imdsTrain, 'DataAugmentation', augmentationOptions);
augImdsValidation = augmentedImageDatastore([256 256 3], imdsVal, 'DataAugmentation', augmentationOptions);

%% Training
net = resnet50(); % Load a pre-trained ResNet-50 model

lgraph = layerGraph(net);

% Update the input layer to accept 256x256x3 images
newInputLayer = imageInputLayer([256, 256, 3], 'Name', 'data');
lgraph = replaceLayer(lgraph, 'input_1', newInputLayer);

% Remove the last layer (fc1000) from the pre-trained ResNet model
lgraph = removeLayers(lgraph, {'fc1000', 'fc1000_softmax', 'ClassificationLayer_fc1000'});

% Modify the network architecture for classification with 17 classes
newLayers = [
    fullyConnectedLayer(17, 'Name', 'fc17');
    softmaxLayer('Name', 'softmax');
    classificationLayer('Name', 'classificationLayer')
];
lgraph = addLayers(lgraph, newLayers);

% Connect the new layers
lgraph = connectLayers(lgraph, 'avg_pool', 'fc17');

options = trainingOptions('sgdm', ...
    'InitialLearnRate', 0.0001, ... % Lower initial learning rate for fine-tuning
    'MaxEpochs', 20, ...
    'Shuffle', 'every-epoch', ...
    'ValidationData', augImdsValidation, ...
    'ValidationFrequency', 10, ...
    'Plots', 'training-progress', ...
    'MiniBatchSize', 32, ...
    'L2Regularization', 1e-4 ... % Add L2 regularization
);


% Train the modified network
net = trainNetwork(augImdsTrain, lgraph, options);
save('classnet', 'net');

%% Evaluation
% Test the network

YPredTest = classify(net, imdsTest);
YTest = imdsTest.Labels;


% Create the confusion matrix
cm = confusionmat(YTest, YPredTest);
cmChart = confusionchart(cm, categories(YTest), 'Normalization', 'row-normalized');
cmChart.Title = 'Normalized Confusion Matrix (%)';

% Calculate precision, recall, and F1-score
precision = diag(cm) ./ sum(cm,2);
recall = diag(cm) ./ sum(cm,1)';
f1Score = 2 * (precision .* recall) ./ (precision + recall);
meanF1Score = mean(f1Score);

% Display the results
fprintf('Mean F1-score: %.2f%%\n', meanF1Score * 100);

%% Function
% Function for custom resizing
function img = customResize(filename)
    img = imread(filename);
    img = imresize(img, [256 256]);
end