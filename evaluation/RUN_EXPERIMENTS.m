% RUN_EXPERIMENTS: Test GP regression approximation 
% methods on a chosen dataset. 
%
% Implemented methods:
% Subset of Data (SoD) -- simply discard some datapoints.
% Local GP [1]         -- cluster input points and run a 
%                         separate GP on each cluster.
% Fully Independent Training Conditional (FITC) [1] -- use 
%                          covariance function approximation
%
% [1] J. Quinonero-Candela, C. R. Rasmussen, 
% "A Unifying View of Sparse Approximate 
% Gaussian Process Regression", JMLR 6 (2005) 1939-1959
%
% Krzysztof Chalupka, University of Edinburgh 
% and California Institute of Technology, 2012
% kjchalup@caltech.edu

%----------------------------------------
% Set up the experiment.
%----------------------------------------
% Choose dataset: SYNTH2, SYNTH8, CHEM or SARCOS. 
%
% To use custom datasets, add loading and preprocessing to loadData.m. 
EXPERIMENT.DATASET = 'PRECIPITATION'; 
EXPERIMENT.DATASET_FOLD = 1;
% Choose the approximation to use: SoD, Local, FITC.
% To use a custom method, create a test[method_name].m script 
% analogous to out testSoD.m etc.
EXPERIMENT.METHOD = 'HSM';

% Number of experiment repetitions.
EXPERIMENT.NUM_TRIALS = 5; 

% For SoD, FITC and Local EXPERIMENT.M is a list of the 
% subset sizes, inducing point set sizes and cluster sizes resp.
%EXPERIMENT.M = [32 64 128 256 512 1024 2048]; 
EXPERIMENT.M = 2:10;

% Additional parameters -- listed in appropriate 
% test[method_name].m script's preamble.
EXPERIMENT.EXTRA = 'separate'; 

% Save raw experiment results here.
EXPERIMENT.RESULTS_DIR = './results'; 

%----------------------------------------
% Run the experiment.
%----------------------------------------
fprintf('Running %d %s experiments on %s dataset.\nThis might take a while...\n', EXPERIMENT.NUM_TRIALS, EXPERIMENT.METHOD, EXPERIMENT.DATASET);
%loadData;
eval(['test' EXPERIMENT.METHOD]);

%----------------------------------------
% Load the results.
% results struct contains the following fields:
% results.msll - each of length(EXPERIMENT.M) columns contains
%                the mean standardized log losses returned by the 
%                EXPERIMENT.NUM_TRIALS  experiment repetitions for each
%                value of EXPERIMENT.M
%                
% results.mse  - same as results.msll, but shows mean squared errors.
% results.hyp_time - same as results.msll, but shows time used to train
%                    the hyperparameters.
% results.train_time - same as results.msll, but shows time used to train
%                      the parameters (the inverted covariance matrix mostly).
% results.test_time - same as results.msll, but shows the test times, i.e.
%                     time to compute predictive variances and means.
% results.hyps      - each of length(EXPERIMENT.M) cells contains 
%                     EXPERIMENT.NUM_TRIALS matrices. Each column
%                     of each matrix contains the hyperparameters
%                     chosen by the approximation. First D rows
%                     contain log(lengthscale),
%                     (D+1)st row contains log(amplitude) and the
%                     last row contains log(noise std dev).
%
% Any additional entries should contain approximation-specific
% information, for example indices of inducing points used, cluster
% structure, etc.
%----------------------------------------------------------------------
%results = eval(['results' EXPERIMENT.METHOD]);
