function [ x, y, xs, smhyp ] = initEnvSM( sd )
%INITENVSM Summary of this function goes here
%   Detailed explanation goes here

    if nargin == 0
        [x, y, xs, smhyp] = initEnv();
    else
        [x, y, xs, smhyp] = initEnv(sd);
    end
    [n, D] = size(x);
    logell = smhyp.cov(1:D);
    lsf2 = smhyp.cov(D+1);
    % Now let's check that the naive and actual implementation agree.
    M = 3;
    V = randn([M, D]);
    %make sure length scale parameters are larger than half of the original ls
    logsigma = log(exp(2*randn([M, D]).^2)+repmat(exp(2*logell)', M, 1)/2)/2;

    smhyp.cov = [logell; reshape(logsigma, [M*D, 1]); ...
        reshape(V, [M*D, 1]); lsf2+(log(2*pi)*D+sum(logell)*2)/4];
end

