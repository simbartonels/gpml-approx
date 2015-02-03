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

XLEN = 10;
YLEN = 8;
MARGIN = 0.1;
MS = 12;
PLOTFILETYPE='pdf';
FILENAME_SUFFIX = '';

RESULTS_DIR = '';
PLOTS_DIR = './plots/'; % Ready plots go here.
DATASETS = {'SYNTH2', 'SYNTH8', 'CHEM', 'SARCOS'}; % Plot data for these datasets only.
METHODS = {'SoD', 'FITC', 'Local', 'Hybrid'}; % Plot data for these methods only.
plot_colors = {'r', 'g', 'b', 'k'}; % At least as many colors as methods plotted. 

MSLL_lims = {[-7 0], [-1.2 1], [-2.9 2.5], [-2.5 1]};
SMSE_lims = {[10^(-6), 0.4], [10^(-0.7), 10^(0.1)], [10^(-1.1), 1.1], [10^(-25), 0.6]};

for dset_id = 1:length(DATASETS)
    dataset = DATASETS{dset_id};
   
    %----------------------------------------
    % Plot MSLL vs hyper-time.
    %----------------------------------------
    figure;
    for method_id = 1:length(METHODS)
        method = METHODS{method_id};
        % Load data.
        load(sprintf('%sresults%s_%s', RESULTS_DIR, method, dataset));
        results = eval(['results' method]);
        hold on;
        plot(results.hyp_time, results.msll, '.', 'Color', plot_colors{method_id}, 'MarkerSize', MS);
        plots{method_id} = plot(mean(results.hyp_time), mean(results.msll), '-', 'Color', plot_colors{method_id});
        xlabel('Hyperparameter training time [s]');
        ylabel('MSLL');
        set(gca, 'xscale', 'log');
        axis tight;
        ylim(MSLL_lims{dset_id});
    end
    legend(cell2mat(plots), METHODS);
    box off;
    kc_format_figure(XLEN, YLEN, MARGIN);
    print(['-d' PLOTFILETYPE] , [PLOTS_DIR dataset '_hyp_MSLL' FILENAME_SUFFIX]);
                    
    %----------------------------------------
    % Plot MSLL vs test time per datapoint.
    %----------------------------------------
    figure;
    for method_id = 1:length(METHODS)
        method = METHODS{method_id};
        % Load data.
        load(sprintf('%sresults%s_%s', RESULTS_DIR, method, dataset));
        results = eval(['results' method]);
        plot(results.test_time/results.N_test, results.msll, '.', 'Color', plot_colors{method_id}, 'MarkerSize', MS);
        hold on;
        plots{method_id} = plot(mean(results.test_time)/results.N_test, mean(results.msll), '-', 'Color', plot_colors{method_id});
        xlabel('Test time per datapoint [s]');
        ylabel('MSLL');
        set(gca, 'xscale', 'log');
        axis tight;
        ylim(MSLL_lims{dset_id});
    end
    legend(cell2mat(plots), METHODS);
    box off;
    kc_format_figure(XLEN, YLEN, MARGIN);
    print(['-d' PLOTFILETYPE], [PLOTS_DIR dataset '_test_MSLL' FILENAME_SUFFIX]);
    
    %----------------------------------------
    % Plot SMSE vs hyper-time.
    %----------------------------------------
    figure;    
    for method_id = 1:length(METHODS)
        method = METHODS{method_id};
        % Load data.
        load(sprintf('%sresults%s_%s', RESULTS_DIR, method, dataset));
        results = eval(['results' method]);                        
        plot(results.hyp_time, results.mse, '.', 'Color', plot_colors{method_id}, 'MarkerSize', MS);
        hold on;
        plots{method_id} = plot(mean(results.hyp_time), mean(results.mse), '-', 'Color', plot_colors{method_id});
        xlabel('Hyperparameter training time [s]');
        ylabel('SMSE');
        set(gca, 'xscale', 'log');
        if dset_id == 1
            set(gca, 'yscale', 'log');
        end
        axis tight;
        ylim(SMSE_lims{dset_id});;
    end
    legend(cell2mat(plots), METHODS);
    box off;
    kc_format_figure(XLEN, YLEN, MARGIN);
    print(['-d' PLOTFILETYPE], [PLOTS_DIR dataset '_hyp_SMSE' FILENAME_SUFFIX]);
    
    %----------------------------------------
    % Plot SMSE vs test time per datapoint.
    %----------------------------------------
    figure;    
    for method_id = 1:length(METHODS)
        method = METHODS{method_id};
        % Load data.
        load(sprintf('%sresults%s_%s', RESULTS_DIR, method, dataset));
        results = eval(['results' method]);                        
        plot(results.test_time/results.N_test, results.mse, '.', 'Color', plot_colors{method_id}, 'MarkerSize', MS);
        hold on;
        plots{method_id} = plot(mean(results.test_time)/results.N_test, mean(results.mse), '-', 'Color', plot_colors{method_id});
        xlabel('Test time per datapoint [s]');
        ylabel('SMSE');
        set(gca, 'xscale', 'log');
        if dset_id == 1
            set(gca, 'yscale', 'log');
        end
        axis tight;
        ylim(SMSE_lims{dset_id});        
    end
    legend(cell2mat(plots), METHODS);
    box off;
    kc_format_figure(XLEN, YLEN, MARGIN);
    print(['-d' PLOTFILETYPE], [PLOTS_DIR dataset '_test_SMSE' FILENAME_SUFFIX]);
    close all;
end
