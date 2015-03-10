function smhyp = FIC_params_to_Multiscale(hyp, D, M, V)
    if nargin < 4
	V = hyp((D+1+1):(D+1+D*M));
    end
    if hyp(size(hyp, 1)) > -Inf, disp('WARNING: FIC and Multiscale differ when noise is greater 0.'); end
    logell = 2 * hyp(1:D);
    f = (log(2*pi)*D+sum(logell))/2;
    f = log(sqrt(prod(exp(logell))*(2*pi)^D));
    lsf = 2 * hyp(D+1)+f;
    logsigma = repmat(logell'-log(2), M, 1);
    smhyp = [logell; reshape(logsigma, [M*D, 1]); ...
        reshape(V, [M*D, 1]); lsf; hyp(size(hyp, 1))];
end
