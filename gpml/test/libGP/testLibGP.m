function testLibGP()
%TESTLIBGP Summary of this function goes here
%   Detailed explanation goes here
%test against full GP
testFullGP();
testHSM();
disp('Test completed succesfully.');
end

function testFullGP()
[trainX, trainY, testX, hyp] = initEnv();
[alpha, L, nlZ, mF, s2F, mFT] = infLibGPmex(trainX, trainY, testX, 'full', 'CovSum (CovSEard, CovNoise)', unwrap(hyp));
[mF_o, s2F_o, ~, ~, nlZ_o, post] = gp(hyp, @infExact, [], {@covSEard}, @likGauss, trainX, trainY, testX);
mFT_o = gp(hyp, @infExact, [], {@covSEard}, @likGauss, trainX, trainY, trainX);

m1 = 'GPML';
m2 = 'LibGP';
checkError(mF_o, mF, m1, m2, 'test mean predictions');
checkError(s2F_o, s2F, m1, m2, 'test variance predictions');
checkError(mFT_o, mFT, m1, m2, 'training mean predictions');
checkError(nlZ_o, nlZ, m1, m2, 'log-likelihood');

end

function testHSM()
[trainX, trainY, testX, hyp] = initEnv();
D = size(trainX, 2);
if D > 2, error('Test assumes D = 2'); end
M = 4;
m = 2;
l = 4*2.2 *ones([1, D])/3

[alpha, L, nlZ, mF, s2F, mFT] = infLibGPmex(trainX, trainY, testX, 'Solin', 'CovSum (CovSEard, CovNoise)', unwrap(hyp), M, 'Solin');

[ J, lambda ] = initHSM( m, D, l );
[mF_o, s2F_o, ~, ~, nlZ_o, post] = gp(hyp, @infExactDegKernel, [], {@covDegenerate, {@degHSM2, m, l, J, lambda}}, @likGauss, trainX, trainY, testX);
mFT_o = gp(hyp, @infExactDegKernel, [], {@covDegenerate, {@degHSM2, m, l, J, lambda}}, @likGauss, trainX, trainY, trainX);
m1 = 'GPML';
m2 = 'LibGP';
checkError(nlZ_o, nlZ, m1, m2, 'log-likelihood');
checkError(mF_o, mF, m1, m2, 'test mean predictions');
checkError(mFT_o, mFT, m1, m2, 'training mean predictions');
checkError(s2F_o, s2F, m1, m2, 'test variance predictions');
end