function [c,ceq,gc,gceq] = unitdisk(x)
    c = -1;
    %c = sum(x.^2) - 1;
    ceq = [ ];

    if nargout > 2
        gc = 2*x;
        gceq = [];
    end
end