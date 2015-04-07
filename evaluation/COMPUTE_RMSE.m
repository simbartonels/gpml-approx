if ~exist('utestY')
EXPERIMENT.PREPROCESS_DATASET = false;
loadData;
stdTrain=std(trainY);
meanTrain=mean(trainY);
utestY = testY;
EXPERIMENT.PREPROCESS_DATASET = true;
loadData;
a = load(sprintf('%s%s%sresults%s_fold%d_M%d', EXPERIMENT.RESULTS_DIR, EXPERIMENT.DATASET, filesep, EXPERIMENT.METHOD, EXPERIMENT.DATASET_FOLD, EXPERIMENT.M));
end      
[alpha, L, nlZ, mF, s2F] = infLibGPmex(trainX, trainY, testX, 'degenerate', 'CovSum (CovSEard, CovNoise)', a.resultsFastFood.hyp_over_time{run_id}(:, end), ...
	EXPERIMENT.M, 'FastFood', EXPERIMENT.SEED{run_id}, '');
rmse = sqrt(mean((utestY-(mF*stdTrain+meanTrain)).^2))
