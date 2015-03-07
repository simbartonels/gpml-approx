function testLibGP()
%TESTLIBGP Summary of this function goes here
%   Detailed explanation goes here
%test against full GP
testFullGP();
testSM();
testHSM();
disp('Test completed succesfully.');
end

function abstractTest(trainX, trainY, testX, hyp, inf, cov, infLibGP, M, bf)
[alpha, L, nlZ, mF, s2F] = infLibGPmex(trainX, trainY, testX, infLibGP, 'CovSum (CovSEard, CovNoise)', unwrap(hyp), M, bf);
[~, ~, ~, mFT, ~] = infLibGPmex(trainX, trainY, trainX, infLibGP, 'CovSum (CovSEard, CovNoise)', unwrap(hyp), M, bf);
[mF_o, s2F_o, ~, ~, nlZ_o, post] = gp(hyp, inf, [], cov, @likGauss, trainX, trainY, testX);
mFT_o = gp(hyp, inf, [], cov, @likGauss, trainX, trainY, trainX);
m1 = 'GPML';
m2 = 'LibGP';
checkError(nlZ_o, nlZ, m1, m2, 'log-likelihood');
checkError(mF_o, mF, m1, m2, 'test mean predictions');
checkError(mFT_o, mFT, m1, m2, 'training mean predictions');
checkError(s2F_o, s2F, m1, m2, 'test variance predictions');
end

function testFullGP()
[trainX, trainY, testX, hyp] = initEnv();
abstractTest(trainX, trainY, testX, hyp, @infExact, {@covSEard}, 'full', 1, '');
end

function testHSM()
[trainX, trainY, testX, hyp] = initEnv();
D = size(trainX, 2);
if D > 2, error('Test assumes D = 2'); end
M = 4;
m = 2;
l = 4*max(trainX)/3
[ J, lambda ] = initHSM( m, D, l );
abstractTest(trainX, trainY, testX, hyp, @infExactDegKernel, {@covDegenerate, {@degHSM2, m, l, J, lambda}}, 'Solin', M, ' ');
end

function testSM()
M = 10;
[trainX, trainY, testX, hyp] = initEnvSM(M);
abstractTest(trainX, trainY, testX, hyp, @infFITC, {@covSM, M}, 'FIC', M, 'SparseMultiScaleGP');
end