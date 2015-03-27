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

me = mfilename;                                            % what is my filename
mydir = which(me); mydir = mydir(1:end-2-numel(me));        % where am I located
RESULTS_DIR = [mydir, 'results', filesep];
PLOTS_DIR = [mydir, 'plots', filesep]; % Ready plots go here.
DATASETS = {'PRECIPITATION'} % Plot data for these datasets only.
METHODS = {'HSM', 'FastFood', 'SoD', 'FullSE'}%, 'FICfixed'} % Plot data for these methods only.
Ms = {2048, 2048, 1940, 1500} %, 1000}

plot_colors = {'r', 'g', 'b', 'k', 'c'}; % At least as many colors as methods 
                                    % plotted. 
PLOTFILETYPE='pdf';
FILENAME_SUFFIX = '_unedited';

fold = '4'

figure, close
for dset_id = 1:length(DATASETS)
    dataset = DATASETS{dset_id};
    for method_id = 1:length(METHODS)
        method = METHODS{method_id};
        % Load data.
        load(sprintf('%s%s%sresults%s_fold%s_M%d', RESULTS_DIR, dataset, filesep, method, fold, Ms{method_id}));        
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
