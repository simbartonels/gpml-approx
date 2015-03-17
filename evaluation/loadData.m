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
rng('default');
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
    
elseif strcmp(EXPERIMENT.DATASET, 'DEBUG')
    n = 42;
    D = 3;
    n_test = 10;
    trainX = randn([n, D]);
    trainY = randn([n, 1]);
    testX = randn([n_test, D]);
    testY = randn([n_test, 1]);
elseif strcmp(EXPERIMENT.DATASET, 'CT_SLICES')
    disp('Loading file...');
    ctslices = csvread('CT_SLICES/slice_localization_data.csv', 1);
    n = 4280;
    D = 384;
    trainX = ctslices(1:n, 2:D); %leave out Patient ID
    trainY = ctslices(1:n, D+2);
    n_test = size(ctslices, 1) - n;
    testX = ctslices(n+1:n+n_test, 2:D);
    testY = ctslices(n+1:n+n_test, D+2);
    clear ctsclices;
elseif strcmp(EXPERIMENT.DATASET, 'CPU')
    cpuset = load('CPU/cpu.mat');
    trainX = cpuset.Xtrain';
    trainY = cpuset.ytrain';
    testX = cpuset.Xtest';
    testY = cpuset.ytest';
    [n, D] = size(trainX);
    n_test = size(testY);
elseif strcmp(EXPERIMENT.DATASET, 'PUMADYN')
    puma=load('PUMADYN/pumadyn32nm.mat');
    trainX = puma.X_tr;
    trainY = puma.T_tr;
    testX = puma.X_tst;
    testY = puma.T_tst;
    clear puma;
    [n, D] = size(trainX);
    n_test = size(testX);
elseif strcmp(EXPERIMENT.DATASET, 'PRECIPITATION')
    EXPERIMENT.DATASET_FOLDS = 10;
    prec = load('PRECIPITATION/USprec1.txt');
    stats = load('PRECIPITATION/USprec2.txt');
    y = sum(prec(prec(:,14)==0,2:13),2);
    y = y/100;
    %avgy = mean(y);
    %y = y-avgy;
    x = stats(prec(:,14)==0,2:3);
    restore_seed = randi(32000);
    seed = 0; %just that we always take the same shuffle
    rng(seed);
    p = randperm(size(x, 1));
    x = x(p, :);
    y = y(p);
    rng(restore_seed);
    
    clear prec;
    clear stats;
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
trainYStd(trainYStd == 0) = 1; % we don't want to divide by zero.
stdMatrix  = repmat(std(trainX), n, 1);
stdMatrix(stdMatrix == 0) = 1;
trainX = (trainX - meanMatrix);
trainX = trainX./stdMatrix;
trainY = (trainY - trainYMean);
trainY = trainY./trainYStd;

testX  = (testX-repmat(meanMatrix(1,:), size(testX,1),1));
testX = testX./repmat(stdMatrix(1,:), size(testX,1),1);
testY  = (testY - trainYMean);
testY = testY./trainYStd;

if any(any(isnan(trainX) | isinf(trainX))), error('Training set contains NaN Values!'); end
if any(any(isnan(trainY) | isinf(trainY))), error('Training targets contains NaN Values!'); end
if any(any(isnan(testX) | isinf(testX))), error('Test set contains NaN Values!'); end
if any(any(isnan(testY) | isinf(testY))), error('Training targets contains NaN Values!'); end

% Reseed the rng.
rand('seed', 100*sum(clock));
