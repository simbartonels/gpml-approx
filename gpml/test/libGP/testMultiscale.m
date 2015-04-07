function testMultiscale()
	testMSvsFICbf();
	testMultiscalevsFIC();

end

function testMSvsFICbf()
	M = 20;
        D = 12;
	z = randn([2, D]);
	fichyp = randn([D+1+M*D+1, 1]);
	uvx = bfmex('FIC', 0, M, fichyp, D, z);
	hyp = FIC_params_to_Multiscale(fichyp, D, M);
    f = sum(fichyp(1:D)) %/2;
    lsf = 2*fichyp(D+1)+f;    
    hyp(end-1) = lsf;
    exp(2*fichyp(D+1))
	uvxSM = bfmex('SparseMultiScaleGP', 0, M, hyp, D, z);
	checkError(uvx, exp(lsf)*uvxSM, 'FIC', 'MS', 'phi');

        Kuu = bfmex('FIC', 0, M, fichyp, D);
	L = chol(Kuu);
	KuuSM = bfmex('SparseMultiScaleGP', 0, M, hyp, D);
	LSM = chol(KuuSM);
	checkError(L, exp(lsf)*LSM, 'FIC', 'MS', 'chol(Kuu)');
	a = L\uvx;
	b = LSM\uvxSM;
	checkError(a, b, 'FIC', 'MS', 'chol(Kuu)^-1 phi');
	disp('Basis Function test successful.');

end


function testMultiscalevsFIC()
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
    FIChyp = resultsFIC.hyp_over_time{1}(: , ind);
    %FIChyp(size(FIChyp, 1), ind) = -Inf;
    disp('Setting noise to 0 for FIC and Multiscale compliance.');
    [~, L, ~, mFfic, ~] = infLibGPmex(trainX, trainY, testX, 'FIC', 'CovSum (CovSEard, CovNoise)', FIChyp, M, 'FIC');
diag(L)'
FICtestError = mean((mFfic-testY).^2./varTest)
    hyp = FIC_params_to_Multiscale(FIChyp, D, M);
    f = sum(FIChyp(1:D)) %/2;
    lsf = 2*FIChyp(D+1)+f;    
    hyp(end-1) = lsf;
    hyp(end-1)

	%uvx = bfmex('FIC', 0, M, FIChyp, D, trainX);
        %Kuu = bfmex('FIC', 0, M, FIChyp, D);
	%L = chol(Kuu);
	%a = L\uvx;
	%a(1:5, 1:5)
	%uvxSM = bfmex('SparseMultiScaleGP', 0, M, hyp, D, trainX);
	%KuuSM = bfmex('SparseMultiScaleGP', 0, M, hyp, D);
	%LSM = chol(KuuSM);
	%b = LSM\uvxSM;
	%b(1:5, 1:5)
%return;
    [~, L, ~, mF, ~] = infLibGPmex(trainX, trainY, testX, 'FIC', 'CovSum (CovSEard, CovNoise)', hyp, M, 'SparseMultiScaleGP');
diag(L)'
testError = mean((mF-testY).^2./varTest)

prediction_diff = mean(abs((mFfic - mF)./mFfic))
%if prediction_diff > 1e-5, error('FIC and Multiscale predictions should agree in this setting!'); end
M = 50;
    filename = [mydir 'resultsFIC_fold1_M50.mat'];
    resultsFIC = load(filename);
    resultsFIC = resultsFIC.resultsFIC;
    ind = size(resultsFIC.hyp_over_time{1}, 2)/2;
    FIChyp = resultsFIC.hyp_over_time{1}(:, ind);
    %FIChyp(end) = -10;
    %FIChyp = log(abs(randn(size(FIChyp)))/2);
    %FIChyp = zeros(size(FIChyp))-abs(randn(size(FIChyp)));
    disp('Setting noise to 0 for FIC and Multiscale compliance.');
    [~, L, ~, mFfic, ~] = infLibGPmex(trainX, trainY, testX, 'FIC', 'CovSum (CovSEard, CovNoise)', FIChyp, M, 'FIC');
FICtestError = mean((mFfic-testY).^2./varTest)
    hyp = FIC_params_to_Multiscale(FIChyp, D, M);
    %hyp(end-1) = FIChyp(D+1);
f = sum(FIChyp(1:D)) %/2;
    lsf = 2*FIChyp(D+1)+f;    
    hyp(end-1) = lsf;
    
    diag(L)'
    ind_noise_factor = 1e-6;%*sqrt(prod(exp(logell))*(2*pi)^D);
    [alpha, L, ~, mF, ~] = infLibGPmex(trainX, trainY, testX, 'FIC', 'CovSum (CovSEard, CovNoise)', hyp, M, 'SparseMultiScaleGP', 0, ind_noise_factor);
diag(L)'
exp(hyp(end-1))
prod(2*exp(hyp(1:D)))
testError = mean((mF-testY).^2./varTest)


prediction_diff = mean(abs((mFfic - mF)./mFfic))
if prediction_diff > 1e-5, error('FIC and Multiscale predictions should agree in this setting!'); end
end
