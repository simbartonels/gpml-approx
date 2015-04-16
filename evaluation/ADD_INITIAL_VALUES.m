method = 'Multiscale';
DATASET = 'PUMADYN';
M = 75;

addpath(genpath('../gpml'));
addpath(genpath('./methods'));
addpath('../project/sod');
addpath(genpath('../project'));
disp('Adding KCenterClustering path');
addpath(genpath('../figtree-0.9.3/matlab'));


for fold=1:1
DATASET_FOLD = fold;
resultVarName = sprintf('results%s', method);
results_file = sprintf('%s%s%s%s%s_fold%d_M%d.mat', ['results' filesep], '', DATASET, filesep, resultVarName, DATASET_FOLD, M)
if exist(results_file, 'file') == 2
	disp('WARNING: An experiment results file already exists for this configuration. Attempting continuation of the experiment.');
	load(results_file);
	resultOut = eval(resultVarName);
    trials_old = size(resultOut.('hyp_time'), 2);
	oldEXPERIMENT = resultOut.EXPERIMENT;
    EXPERIMENT = oldEXPERIMENT;
	if ~isfield(EXPERIMENT, 'PREPROCESS_DATASET')
		EXPERIMENT.PREPROCESS_DATASET = true;
    end 
    loadData;
    varTest=var(testY);
    meanTest=mean(testY);
    varTrain=var(trainY);
    meanTrain=mean(trainY);

    if isfield(EXPERIMENT, 'LAST_HYPERS')
        EXPERIMENT = rmfield(EXPERIMENT, 'LAST_HYPERS');
    end
    EXPERIMENT.NUM_HYPER_OPT_ITERATIONS = 0;
    msmode = false;
    if strcmp(method, 'Multiscale'), msmode = true; end
	for trial_id = 1:trials_old
		initial_grad_norm = resultOut.grad_norms{trial_id}(1);
        msrestart = false;
        if msmode
               filename = sprintf('%s%s%sresultsFIC_fold%d_M%d.mat', ...
                EXPERIMENT.RESULTS_DIR, EXPERIMENT.DATASET, filesep, EXPERIMENT.DATASET_FOLD, EXPERIMENT.M)
                resultsFIC = load(filename);
                resultsFIC = resultsFIC.resultsFIC;
                FIChyp = resultsFIC.hyp_over_time{trial_id};
                ind = floor(size(FIChyp, 2)/2);
            %    ind = 25 %TODO: remove
                FIChyp = FIChyp(:, ind);
                time_offset = resultsFIC.hyp_time{trial_id}(ind);
                mstime_offset = resultOut.hyp_time{trial_id}(1);
                if time_offset < mstime_offset
                    msrestart = true; 
                elseif mstime_offset == 0
                    resultOut.hyp_time{trial_id}(1) = time_offset;
                    resultOut.EXPERIMENT = oldEXPERIMENT;
                    eval(sprintf('%s=resultOut;', resultVarName));
                    save(results_file, resultVarName);
                end
        end
        if initial_grad_norm > -1 || msrestart
                [~, times, theta_over_time, mF, s2F, nlZ, gradNorms, ~] = feval(EXPERIMENT.METHOD, EXPERIMENT, trainX, trainY, testX, trial_id);
                resultOut.('hyp_time'){trial_id} = [times(1), resultOut.('hyp_time'){trial_id}];
                resultOut.('hyp_over_time'){trial_id} = [theta_over_time(:, 1), resultOut.('hyp_over_time'){trial_id}];
                resultOut.('msll'){trial_id}= [mnlp(mF(:, 1),testY,s2F(:, 1), meanTest, varTest), resultOut.('msll'){trial_id}];
                resultOut.('mse'){trial_id}  = [mse(mF(:, 1),testY, meanTest, varTest), resultOut.('mse'){trial_id}];
		    	resultOut.('llh'){trial_id} = [nlZ(1), resultOut.('llh'){trial_id}];
				resultOut.('grad_norms'){trial_id} = [gradNorms(1), resultOut.('grad_norms'){trial_id}];
        else
            fprintf('Nothing to be done for trial %d\n', trial_id);
        end
    end
    resultOut.EXPERIMENT = oldEXPERIMENT;
    eval(sprintf('%s=resultOut;', resultVarName))
    save(results_file, resultVarName);
end
end