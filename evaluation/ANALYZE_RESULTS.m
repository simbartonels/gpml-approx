% ANALYZE_DATA: Plot standardized mean squared error (SMSE) and 
% mean standardized log loss (MSLL) as a function of time for all 
% given datasets/methods.
% 
% This script should be run after RUN_EXPERIMENTS completes.
% See RUN_EXPERIMENTS for an explanation of the data format etc. 
% The evaluations/data/ directory contains by default the datafiles 
% used in our JMLR publication [1]. An easy way to compare your 
% approximation's performance with our results is to run RUN_EXPERIMENTS
% with your method, and place the results in the same directory. Then,
% simply run this script with obvious modifications (see code below).
%
% Our own results are attached, for comparison. See datafiles under
% ./Chalupka_Williams_Murray/results[METHOD]_[DATASET].m.
%
% NOTE: This provides only basic plotting, you'll probably need to set
% xlims, and ylims manually to get reasonable plots.
%
% [1] K. J. Chalupka, C. K. I. Williams, I. Murray, "A Framework for 
% Evaluating Approximation Methods for Gaussian Process Regression", 
% JMLR 2012 (submitted)
%
% Krzysztof Chalupka, University of Edinburgh 
% and California Institute of Technology, 2012
% kjchalup@caltech.edu
clear;
me = mfilename;                                            % what is my filename
mydir = which(me); mydir = mydir(1:end-2-numel(me));        % where am I located
RESULTS_DIR = [mydir, 'results', filesep];
PLOTS_DIR = [mydir, 'plots', filesep]; % Ready plots go here.
DATASETS = {'PRECIPITATION'} % Plot data for these datasets only.
METHODS = {'HSM', 'FastFood', 'SoD', 'FIC'}%, 'FICfixed'} % Plot data for these methods only.
Ms = {2048, 2048, 1940, 1500, 1500} %, 1000}
%DATASETS = {'PUMADYN'}
%METHODS = {'FIC', 'FastFood', 'SoD'}%, 'Multiscale'}
%Ms = {75, 198, 85, 75}
%METHODS = {'FIC', 'Multiscale', 'MultiscaleFIC'};
%Ms = {50, 50, 25}
%DATASETS = {'CPU'}
%METHODS = {'MultiscaleFIC', 'FIC'}%, 'FastFood', 'FIC'}
%Ms = {20, 20} 
plot_colors = {'r', 'g', 'b', 'k', 'c'}; % At least as many colors as methods 
                                    % plotted. 
PLOTFILETYPE='pdf';
FILENAME_SUFFIX = '_unedited';

fold = '1'

figure, close
for dset_id = 1:length(DATASETS)
    dataset = DATASETS{dset_id};
    dataset_loaded = false;
    for method_id = 1:length(METHODS)
        method = METHODS{method_id};
        % Load data.
	results_file = sprintf('%s%s%sresults%s_fold%s_M%d', RESULTS_DIR, dataset, filesep, method, fold, Ms{method_id});
        load(results_file);
	method = METHODS{method_id};
	resultVarName = sprintf('results%s', method);
        results = eval(resultVarName);
	if ~isfield(results, 'pred_time') & ~strcmp(method, 'FullSE')
		EXPERIMENT = results.EXPERIMENT;
		if ~dataset_loaded
			EXPERIMENT.PREPROCESS_DATASET = true;
			loadData;
			dataset_loaded = true;
		end
		if strcmp(method, 'FastFood')
			gpname = 'degenerate';
			bfname = 'FastFood';
			rmfield(EXPERIMENT, 'EXTRA');
		elseif strcmp(method, 'HSM')
			gpname = 'degenerate';
			bfname = 'FastFood';
		elseif strcmp(method, 'SoD')
			gpname = 'full';
			bfname = '';
			trainX = trainX(EXPERIMENT.SOD{1}, :);
			trainY = trainY(EXPERIMENT.SOD{1}, :);
		elseif strcmp(method, 'FIC')
			gpname = 'OptFIC';
			bfname = 'FIC';
		elseif strcmp(method, 'Multiscale')
			gpname = 'OptMultiscale';
			bfname = 'SparseMultiScaleGP';
		else
			error('Unkown method name: %s', method);
		end
		if isfield(EXPERIMENT, 'EXTRA')
	                [~, ~, ~, ~, ~, t] = infLibGPmex(trainX, trainY, testX, gpname, 'CovSum (CovSEard, CovNoise)', results.hyp_over_time{1}(:, 1), EXPERIMENT.M, bfname, EXPERIMENT.SEED{1}, EXPERIMENT.EXTRA)
		else
			[~, ~, ~, ~, ~, t] = infLibGPmex(trainX, trainY, testX, gpname, 'CovSum (CovSEard, CovNoise)', results.hyp_over_time{1}(:, 1), EXPERIMENT.M, bfname, EXPERIMENT.SEED{1})
		end
		results.pred_time = t;
		%avg_pred_time = t / size(testX, 1)
		eval(sprintf('%s=results;', resultVarName));
        	%save(results_file, resultVarName);
	end
	prediction_time = results.pred_time
    end
end
for dset_id = 1:length(DATASETS)
    dataset = DATASETS{dset_id};
    figure 'visible' 'off';
    %----------------------------------------
    % Plot MSLL vs hyper-time.
    %----------------------------------------
    for method_id = 1:length(METHODS)
        method = METHODS{method_id};
        results = eval(sprintf('results%s', method));
        hold on;
        x_value = results.hyp_time;
        y_value = results.msll;
        myplot;
        xlabel('Hyperparameter training time [s]');
        ylabel('MSLL');
        %set(gca, 'xscale', 'log');
    end
    legend(cell2mat(plots), METHODS);
    print(['-d' PLOTFILETYPE] , [PLOTS_DIR dataset '_hyp_MSLL' FILENAME_SUFFIX]);

    %----------------------------------------
    % Plot llh and TSMSE vs hyper-time.
    %----------------------------------------
    figure 'visible' 'off';
    for method_id = 1:length(METHODS)
        method = METHODS{method_id};
        results = eval(sprintf('results%s', method));
        hold on;
        x_value = results.hyp_time;
        y_value = results.llh;
        myplot;
        xlabel('Hyperparameter training time [s]');
        ylabel('-LLH');
        %set(gca, 'xscale', 'log');
    end
    legend(cell2mat(plots), METHODS);
    print(['-d' PLOTFILETYPE] , [PLOTS_DIR dataset '_hyp_LLH' FILENAME_SUFFIX]);
    
    %----------------------------------------
    % Plot SMSE vs hyper-time.
    %----------------------------------------
    figure('visible', 'off');
    for method_id = 1:length(METHODS)
        method = METHODS{method_id};
        results = eval(sprintf('results%s', method));
        hold on;
        x_value = results.hyp_time; %(trial_id, :);
        y_value = results.mse; %(trial_id, :);
        myplot;
        xlabel('Hyperparameter training time [s]');
        ylabel('SMSE');
        %set(gca, 'xscale', 'log');
        %set(gca, 'yscale', 'log');
    end
    legend(cell2mat(plots), METHODS);
    print(['-d' PLOTFILETYPE], [PLOTS_DIR dataset '_hyp_SMSE' FILENAME_SUFFIX]);

    close all;
end
