function fToyExp(D, n, M, Mff, sn2, sd, c)
rng(sd);
trainX = randn([n, D]);
%testX = randn([n, D]);
testX = trainX + 1e-5 * randn([n, D]);

testX = trainX;

meanMatrix = repmat(mean(trainX), n, 1);
stdMatrix  = repmat(std(trainX), n, 1);
stdMatrix(stdMatrix == 0) = 1;
trainX = (trainX - meanMatrix);
trainX = trainX./stdMatrix;
testX  = (testX-repmat(meanMatrix(1,:), size(testX,1),1));
testX = testX./repmat(stdMatrix(1,:), size(testX,1),1);

sod = indPoints(trainX, M, 'c');
U = trainX(sod, :);
U = reshape(U, [M*D, 1]);

%this is worse
ell = ones([D, 1])*M/n;
Uell = ones([M*D, 1])*n/M-repmat(ell, [M, 1])/2;
smhyp=[log(ell); log(Uell); U; D*log(2*pi)/2; sn2];
[e1, e2] = toyExpTwo(sd, trainX, smhyp, Mff, sn2, testX, c)

%this is better
ell = ones([D, 1]);
Uell = ones([M*D, 1])-repmat(ell, [M, 1])/2;
smhyp=[log(ell); log(Uell); U; D*log(2*pi)/2; sn2];
[e1, e2] = toyExpTwo(sd, trainX, smhyp, Mff, sn2, testX, c)

[~, ~, trainY] = toyExpTwo(sd, trainX, smhyp, Mff, sn2, testX, c);
[sortedValues,sortIndex] = sort(trainY,'descend');  %# Sort the values in
                                                  %#   descending order
maxIndex = sortIndex(1:M);
U = reshape(trainX(maxIndex, :), [M*D, 1]);
smhyp=[log(ell); log(Uell); U; D*log(2*pi)/2; sn2];
[e1, e2] = toyExpTwo(sd, trainX, smhyp, Mff, sn2, testX, c)

[inds, ~] = indPoints(trainX, M, 'c');
U = zeros(M, D);
for m=1:M
        tempY = trainY;
	tempY(inds ~= m) = 0;
	[~, ind] = max(abs(tempY));
	U(m, :) = trainX(ind, :);
end
U = reshape(U, [M*D, 1]);
smhyp=[log(ell); log(Uell); U; D*log(2*pi)/2; sn2];
[~, e2] = toyExpTwo(sd, trainX, smhyp, Mff, sn2, testX, c)

gpmlhyp.cov = smhyp(1:end-1);
gpmlhyp.lik = smhyp(end);
[nlZ, dnlZ] = gp(gpmlhyp, @infFITC, [], {@covSM, M}, @likGauss, trainX, trainY);
%dnlZ.cov
%dnlZ.lik
nlZ

ell = ell / 2;
smhyp=[log(ell); log(Uell); U; D*log(2*pi)/2; sn2];
gpmlhyp.cov = smhyp(1:end-1);
gpmlhyp.lik = smhyp(end);
[nlZ, dnlZ] = gp(gpmlhyp, @infFITC, [], {@covSM, M}, @likGauss, trainX, trainY);
%dnlZ.cov
dnlZ.lik
[~, e2] = toyExpTwo(sd, trainX, smhyp, Mff, sn2, testX, c)
nlZ

error('foo');

iters = 3;
[~, hyps, ~, ~, ~]= rpropmex(sd, iters, trainX, trainY, testX, 'OptMultiscale', 'CovSum ( CovSEard, CovNoise)', smhyp, M, 'SparseMultiScaleGP', c);
[e1, e2] = toyExpTwo(sd, trainX, smhyp, Mff, sn2, testX, c)
[e1, e2] = toyExpTwo(sd, trainX, hyps(:, end), Mff, sn2, testX, c)
%(hyps(1:(D+M*D), end)-smhyp(1:(D+M*D)))'
hyps(end, end)
end
