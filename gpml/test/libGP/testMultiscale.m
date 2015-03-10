function testMultiscale()
%tests on a real dataset if FIC and Multiscale agree.
me = mfilename;                                            % what is my filename
mydir = which(me); mydir = mydir(1:end-2-numel(me));
mydir = [mydir 'misc' filesep];
    filename = [mydir 'resultsFIC_PUMADYN_fold4.mat'];
    resultsFIC = load(filename);
    resultsFIC = resultsFIC.resultsFIC;
    puma=load([mydir 'pumadyn32nm.mat']);
    trainX = puma.X_tr;
    trainY = puma.T_tr;
    testX = puma.X_tst;
    testY = puma.T_tst;
    clear puma;
    [n, D] = size(trainX);
    n_test = size(testX);

meanMatrix = repmat(mean(trainX), n, 1);
trainYMean = mean(trainY);
trainYStd  = std(trainY);
stdMatrix  = repmat(std(trainX), n, 1);
trainX = (trainX - meanMatrix);
trainX = trainX./stdMatrix;
trainY = (trainY - trainYMean);
trainY = trainY./trainYStd;

testX  = (testX-repmat(meanMatrix(1,:), size(testX,1),1));
testX = testX./repmat(stdMatrix(1,:), size(testX,1),1);
testY  = (testY - trainYMean);
testY = testY./trainYStd;

varTest=var(testY);
meanTest=mean(testY);
varTrain=var(trainY);
meanTrain=mean(trainY);


    M = 20;
    ind = 25;
    FIChyp = resultsFIC.hyp_over_time{1};
    FIChyp(size(FIChyp, 1), ind) = -Inf;
    disp('Setting noise to 0 for FIC and Multiscale compliance.');
    [~, ~, ~, mFfic, ~] = infLibGPmex(trainX, trainY, testX, 'FIC', 'CovSum (CovSEard, CovNoise)', FIChyp(: , ind), M, 'FIC');
FICtestError = mean((mFfic-testY).^2./varTest)
    hyp = FIC_params_to_Multiscale(FIChyp(:, ind), D, M);
    [~, ~, ~, mF, ~] = infLibGPmex(trainX, trainY, testX, 'FIC', 'CovSum (CovSEard, CovNoise)', hyp, M, 'SparseMultiScaleGP');
testError = mean((mF-testY).^2./varTest)

prediction_diff = mean(abs((mFfic - mF)./mFfic))
if prediction_diff > 1e-5, error('FIC and Multiscale predictions should agree in this setting!'); end
end
