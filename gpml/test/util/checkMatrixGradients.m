function checkMatrixGradients( fun, x0, n, m )
%CHECKMATRIXGRADIENTS Checks that the gradients of a matrix are correct.
    options = optimoptions(@fmincon,'Algorithm','interior-point',...
        'DerivativeCheck','on','GradObj','on', 'MaxFunEvals', 1);
    k = floor(rand(1)*n+1);
    l = floor(rand(1)*m+1);
    % actual optimization function
    actoptfunc = @(x) optfunc(x, fun, k, l);
    %derivative check
    %try
    fmincon(actoptfunc,...
                   x0,[],[],[],[],[],[],@unitdisk,options);
    %catch
    %    error('Gradientcheck in checkMatrixGradient failed');
    %end
end

function [y, dy] = optfunc(x, fun, k, l)
    [K, dK] = fun(x);
    y = K(k, l);
    dy = dK(k, l);
end