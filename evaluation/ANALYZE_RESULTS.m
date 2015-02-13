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

RESULTS_DIR = './Chalupka_Williams_Murray_Results/'; %
me = mfilename;                                            % what is my filename
mydir = which(me); mydir = mydir(1:end-2-numel(me));        % where am I located
RESULTS_DIR = [mydir, 'results\'];
PLOTS_DIR = [mydir, 'plots\']; % Ready plots go here.
DATASETS = {'SYNTH2'}; % Plot data for these datasets only.
METHODS = {'HSM'}; % Plot data for these methods only.

plot_colors = {'r', 'g', 'b', 'k'}; % At least as many colors as methods 
                                    % plotted. 
PLOTFILETYPE='pdf';
FILENAME_SUFFIX = '_unedited';

for dset_id = 1:length(DATASETS)
    dataset = DATASETS{dset_id};
   
    %----------------------------------------
    % Plot MSLL vs hyper-time.
    %----------------------------------------
    figure;
    for method_id = 1:length(METHODS)
        method = METHODS{method_id};
        % Load data.
        load(sprintf('%sresults%s_%s_fold1', RESULTS_DIR, method, dataset));
        results = eval(sprintf('results%s', method));
        hold on;
        plot(results.hyp_time, results.msll, '.', 'Color', plot_colors{method_id});
        plots{method_id} = plot(mean(results.hyp_time), mean(results.msll), '-', 'Color', plot_colors{method_id});
        xlabel('Hyperparameter training time [s]');
        ylabel('MSLL');
        set(gca, 'xscale', 'log');
    end
    legend(cell2mat(plots), METHODS);
    print(['-d' PLOTFILETYPE] , [PLOTS_DIR dataset '_hyp_MSLL' FILENAME_SUFFIX]);
                    
    %----------------------------------------
    % Plot MSLL vs test time per datapoint.
    %----------------------------------------
    figure;
    for method_id = 1:length(METHODS)
        method = METHODS{method_id};
        % Load data.
        load(sprintf('%sresults%s_%s_fold1', RESULTS_DIR, method, dataset));
        results = eval(sprintf('results%s', method));
        plot(results.test_time/results.N_test, results.msll, '.', 'Color', plot_colors{method_id});
        hold on;
        plots{method_id} = plot(mean(results.test_time)/results.N_test, mean(results.msll), '-', 'Color', plot_colors{method_id});
        xlabel('Test time per datapoint [s]');
        ylabel('MSLL');
        set(gca, 'xscale', 'log');
    end
    legend(cell2mat(plots), METHODS);
    print(['-d' PLOTFILETYPE], [PLOTS_DIR dataset '_test_MSLL' FILENAME_SUFFIX]);
    
    %----------------------------------------
    % Plot SMSE vs hyper-time.
    %----------------------------------------
    figure;    
    for method_id = 1:length(METHODS)
        method = METHODS{method_id};
        % Load data.
        load(sprintf('%sresults%s_%s_fold1', RESULTS_DIR, method, dataset));
        results = eval(sprintf('results%s', method));
        plot(results.hyp_time, results.mse, '.', 'Color', plot_colors{method_id});
        hold on;
        plots{method_id} = plot(mean(results.hyp_time), mean(results.mse), '-', 'Color', plot_colors{method_id});
        xlabel('Hyperparameter training time [s]');
        ylabel('SMSE');
        set(gca, 'xscale', 'log');
        set(gca, 'yscale', 'log');
    end
    legend(cell2mat(plots), METHODS);
    print(['-d' PLOTFILETYPE], [PLOTS_DIR dataset '_hyp_SMSE' FILENAME_SUFFIX]);
    
    %----------------------------------------
    % Plot SMSE vs test time per datapoint.
    %----------------------------------------
    figure;    
    for method_id = 1:length(METHODS)
        method = METHODS{method_id};
        % Load data.
        load(sprintf('%sresults%s_%s_fold1', RESULTS_DIR, method, dataset));
        results = eval(sprintf('results%s', method));
        plot(results.test_time/results.N_test, results.mse, '.', 'Color', plot_colors{method_id});
        hold on;
        plots{method_id} = plot(mean(results.test_time)/results.N_test, mean(results.mse), '-', 'Color', plot_colors{method_id});
        xlabel('Test time per datapoint [s]');
        ylabel('SMSE');
        set(gca, 'xscale', 'log');
        set(gca, 'yscale', 'log');
    end
    legend(cell2mat(plots), METHODS);
    print(['-d' PLOTFILETYPE], [PLOTS_DIR dataset '_test_SMSE' FILENAME_SUFFIX]);
    close all;
end
