% loadData: Load data for an experiment, in a format
% compatible with test[method_name].m scripts.
%
% REQUIRES:
% The script uses the EXPERIMENT structure containing
% experiment parameters (see RUN_TESTS.m).
%
% RETURNS:
% Given a dataset name, the script should load the 
% following matrices into memory :
%
% D - scalar input dimensionality
% n - scalar number of training points
% trainX - n times D, n datapoints of dimensionality D
% trainY - n times 1 (the outputs are scalars)
% testX, testY - similar to trainX, trainY
%
% Krzysztof Chalupka, University of Edinburgh 
% and California Institute of Technology, 2012
% kjchalup@caltech.edu



if ~exist('EXPERIMENT') 
    error('EXPERIMENT structure not available.\n');
end
if ~isfield(EXPERIMENT, 'DATASET')
    error('DATASET field in EXPERIMENT not set.\n');
end

EXPERIMENT.DATA_SET_FOLDS = 1;

%----------------------------------------
% MODIFY THIS TO
% ADD YOUR OWN DATASET PREPROCESSING.
%----------------------------------------
if strcmp(EXPERIMENT.DATASET, 'SYNTH2')
    %----------------------------------------
    % SYNTH2 
    %----------------------------------------
    load('SYNTH/T02');
    ids=randperm(size(x,1)); 
    n=ceil(length(ids)/2);
    D=2;
    trainX=x(ids(1:n),:);
    trainY=y(ids(1:n),:);
    testX=x(ids(end-n+1:end),:);
    testY=y(ids(end-n+1:end),:);

elseif strcmp(EXPERIMENT.DATASET, 'SYNTH8')
    %----------------------------------------
    % SYNTH8
    %----------------------------------------
    load('SYNTH/T08');
    load('SYNTH/T00'); % Loads Gaussian noise.
    ids=randperm(size(x,1));
    n = ceil(length(ids)/2);
    D=8;
    trainX=x(ids(1:n),:);
    trainY=y(ids(1:n),:);
    testX=x(ids(end-n+1:end),:);
    testY=y(ids(end-n+1:end),:);
    trainY = trainY + sqrt(0.001)*noise(ids(1:n)); % Add noise to the data. 
    testY = testY + sqrt(0.001)*noise(ids(end-n+1:end));
    
elseif strcmp(EXPERIMENT.DATASET, 'CHEM')
    %----------------------------------------
    % CHEM 
    %----------------------------------------
    load('CHEM/vinylBromide.txt');
    x=vinylBromide(:,1:15);
    y=vinylBromide(:,16);
    rand('seed', 0);
    ids=randperm(size(x,1));
    n=ceil(length(ids)/2); 
    D=15;
    trainX=x(ids(1:n),:);
    trainY=y(ids(1:n),:);
    testX=x(ids(end-n+1:end),:);
    testY=y(ids(end-n+1:end),:);

elseif strcmp(EXPERIMENT.DATASET,'SARCOS')
    %----------------------------------------
    % SARCOS
    %----------------------------------------
    train=load('SARCOS/sarcos_inv.mat')
    train=train.sarcos_inv;
    test=load('SARCOS/sarcos_inv_test.mat')
    test=test.sarcos_inv_test;
    n=size(train,1);
    ids = randperm(n);
    D=21;
    trainX = train(1:n,1:D); trainY = train(1:n,D+1);
    testX  = test(:,1:D); testY = test(:,D+1);

elseif strcmp(EXPERIMENT.DATASET, 'PRECIPITATION')
    EXPERIMENT.DATA_SET_FOLDS = 10;
    prec = load('PRECIPITATION\\USprec1.txt');
    stats = load('PRECIPITATION\\USprec2.txt');
    y = sum(prec(prec(:,14)==0,2:13),2);
    y = y/100;
    %avgy = mean(y);
    %y = y-avgy;
    x = stats(prec(:,14)==0,2:3);
    % In http://lib.tkk.fi/Dipl/2010/urn100140.pdf a subset of size 223 is used as validation set.
    % Probably Solin did the same.
    [n, D] = size(x);
    k = EXPERIMENT.DATASET_FOLD;
    a = (k-1)*floor(n/10) + 1;
    b = k*floor(n/10);
    testX = x(a:b, :);
    testY = y(a:b);
    trainX = x([1:(a-1),(b+1):n], :);
    trainY = y([1:(a-1),(b+1):n]);
    n = size(trainX, 1);
end

%------------------------------------------------------------------------
% DON'T MODIFY THIS.
% Normalize data to zero mean, variance one. 
%------------------------------------------------------------------------
meanMatrix = repmat(mean(trainX), n, 1);
trainYMean = mean(trainY);
trainYStd  = std(trainY);
stdMatrix  = repmat(std(trainX), n, 1);
trainX = (trainX - meanMatrix);
trainX = trainX./stdMatrix;
trainY = (trainY - trainYMean)./trainYStd;

testX  = (testX-repmat(meanMatrix(1,:), size(testX,1),1))./repmat(stdMatrix(1,:), size(testX,1),1);
testY  = (testY - trainYMean)./trainYStd;

% Reseed the rng.
rand('seed', 100*sum(clock));
