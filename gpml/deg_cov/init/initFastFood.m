function [s, gpi, b] = initFastFood(m, D, hyp)
% Initializes all the necessary matrices for Fast Food.
% m - the number of blocks to use, i.e. m*D is the total number of basis
% functions.
% D - input dimensionality
% hyp - hyper parameters of the isotropic Squared Exponential kernel.
    D = 2^nextpow2(D);
    ell = exp(hyp(1));                                 % characteristic length scale
    sf2 = exp(2*hyp(2));                               % signal variance
    
%     r = randn([m*D, 1]);%*ell; %TODO: plus something with ell and 2pi
%     A = 2*pi^(D/2)/factorial(D/2);
%     s = r.^(D-1).*exp(-r.^2/2)/(A*sqrt((2*pi)^D));
%     s = r.^(D-1).*exp(-r.^2/2);
    s = zeros([m*D, 1]);
    b = 2*(randi([0 1], [m*D,1])-0.5);
    randpi = zeros([m*D, 1]);
    for j = 1:m
        randpi((1+D*(j-1)):D*j) = randperm(D);
        for d = 1:D
            for l = 1:D
                s(D*(j-1)+d) = s(D*(j-1)+d)+randn(1)^2;
            end
        end
    end
    s = sqrt(s);
    g = randn([m*D, 1]);
    gpi = g.*randpi(:);
    for j = 1:m
        idx = (1+D*(j-1)):D*j;
        no = norm(gpi(idx, 1), 'fro');
        %TODO: could be a typo
        %no = sqrt(no);
        s(idx)=s(idx)/no;
    end
    %the factor from equation (7) without ls
    s = s/sqrt(D); 
end