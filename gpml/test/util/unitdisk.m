function [c,ceq,gc,gceq] = unitdisk(x)
    c = sum(x.^2) - 1;
    ceq = [ ];

    if nargout > 2
        gc = 2*x;
        gceq = [];
    end
end