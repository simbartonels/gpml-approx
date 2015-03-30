D = 2;
n = 800;
Mms = 800;
Mff = 800;

addpath(genpath('../gpml'));
addpath(genpath('./methods'));
addpath('../project/sod');
addpath(genpath('../project'));
disp('Adding KCenterClustering path');
addpath(genpath('../figtree-0.9.3/matlab'));

maxNumCompThreads(1);
startup

trainX = randn([n, D]);
sod = indPoints(trainX, Mms, 'c');
U = trainX(sod, :);
U = reshape(U, [Mms*D, 1]);
[e1, e2] = toyExpTwo(0, trainX, U, Mff)
