D = 2;
n = 80;
Mms = 80;
Mff = 80;
sn2 = log(0.05);
addpath(genpath('../gpml'));
addpath(genpath('./methods'));
addpath('../project/sod');
addpath(genpath('../project'));
disp('Adding KCenterClustering path');
addpath(genpath('../figtree-0.9.3/matlab'));

maxNumCompThreads(1);
startup

fToyExp(D, n, Mms, Mff, sn2, 1)
