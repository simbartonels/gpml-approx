function [fx, dx] = optimfunc(hypx, hyp0, inf, mean, cov, lik, x, y)
%OPTIMFUNC Function used for checking gradients.
    if nargout > 1
        [fx, dL] = gp(rewrap(hyp0, hypx), inf, mean, cov, lik, x, y);
        dx = unwrap(dL);
    else
        fx = gp(rewrap(hyp0, hypx), inf, mean, cov, lik, x, y);
    end
end