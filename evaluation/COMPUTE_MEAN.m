clear;
EXPERIMENT.METHOD = 'Multiscale';
EXPERIMENT.DATASET = 'PRECIPITATION';
EXPERIMENT.DATASET_FOLD = 0;
EXPERIMENT.M = 1500;
EXPERIMENT.RESULTS_DIR = './results/'; 
resultVarName = sprintf('results%s', EXPERIMENT.METHOD);
results_file = sprintf('%s%s%s%s%s_fold%d_M%d.mat', EXPERIMENT.RESULTS_DIR, '', EXPERIMENT.DATASET, filesep, resultVarName, EXPERIMENT.DATASET_FOLD, EXPERIMENT.M)
load(results_file);
resultOut = eval(resultVarName);
object = resultOut.msll;
meanvs = zeros([size(object, 2), 1]);
for j = 1:size(object, 2)
    if j ~= 5
    meanvs(j) = object{j}(end);
    end
end
mean(meanvs)