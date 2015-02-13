function [mF, s2F] = HSM_predict(EXPERIMENT, hyp, trainX, trainY, testX, retvals)
    [mF, s2F] = gp(hyp, retvals{1}, [], retvals{2}, retvals{3}, trainX, trainY, testX);
end