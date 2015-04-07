clear;
EXPERIMENT.METHOD = 'SoD';
EXPERIMENT.DATASET = 'PRECIPITATION';
EXPERIMENT.M = 1940;
EXPERIMENT.NUM_TRIALS = 0;
EXPERIMENT.RESULTS_DIR = './results/'; 
resultVarName = sprintf('results%s', EXPERIMENT.METHOD);
folds = 10;
trial_counter = 1;
for d = 1:folds
    results_file = sprintf('%s%s%s%s%s_fold%d_M%d.mat', EXPERIMENT.RESULTS_DIR, '', EXPERIMENT.DATASET, filesep, resultVarName, d, EXPERIMENT.M)
    if exist(results_file, 'file') == 2
        disp('WARNING: An experiment results file already exists for this configuration. Attempting continuation of the experiment.');
        a = load(results_file);
        resultOut = eval(['a.' resultVarName]);
        EXPERIMENT = resultOut.EXPERIMENT;
    else
        error('Result file not found.');
    end
    first_trial_id = 1;
    last_trial_id = size(resultOut.hyp_time, 1);
    m = EXPERIMENT.M;
    for trial_id = first_trial_id:last_trial_id
        resultOutNew.seeds{trial_counter} = resultOut.('seeds'){trial_id};
        resultOutNew.('hyp_time'){trial_counter} = resultOut.hyp_time{trial_id};
        resultOutNew.('hyp_over_time'){trial_counter} = resultOut.hyp_over_time{trial_id};
        resultOutNew.('msll'){trial_counter} = resultOut.msll{trial_id};
        resultOutNew.('mse'){trial_counter}  = resultOut.mse{trial_id};
        resultOutNew.('llh'){trial_counter} = resultOut.llh{trial_id};   
        resultOutNew.('grad_norms'){trial_counter} = resultOut.grad_norms{trial_id};
        EXPERIMENT.LAST_HYPERS{trial_counter} = resultOut.hyp_over_time{trial_id}(:, end);
        resultOutNew.('EXPERIMENT') = EXPERIMENT;
        trial_counter = trial_counter + 1;
    end
end
        EXPERIMENT
        EXPERIMENT.LAST_HYPERS{:}
        resultOutNew
        eval(sprintf('%s=resultOutNew;', resultVarName));
        results_file = sprintf('%s%s%s%s%s_fold%d_M%d.mat', EXPERIMENT.RESULTS_DIR, '', EXPERIMENT.DATASET, filesep, resultVarName, 0, EXPERIMENT.M)
        save(results_file, resultVarName); 
clear EXPERIMENT;
clear resultOut;