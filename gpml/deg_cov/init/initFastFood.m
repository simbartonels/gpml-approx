function [s, gpi, b] = initFastFood(m, D, hyp)
% Initializes all the necessary matrices for Fast Food.
% m - the number of blocks to use, i.e. m*D is the total number of basis
% functions.
% D - input dimensionality
% hyp - hyper parameters of the isotropic Squared Exponential kernel.
    D = 2^nextpow2(D);
    ell = exp(hyp(1));                                 % characteristic length scale
    sf2 = exp(2*hyp(2));                               % signal variance
    
    b = 2*(randi([0 1], [m*D,1])-0.5);
    randpi = randperm(m*D);
    g = randn([m*D, 1]);
    gpi = g.*randpi(:);
    r = randn(1, m)*ell; %TODO: plus something with ell and 2pi
    A = pi^(D/2)/factorial(D/2);
    s = r.^(D-1).*exp(-r.^2/2)/(A*sqrt((2*pi)^D));
    no = zeros(1, m);
    for j = 1:m
        no(j) = sqrt(norm(gpi((1+D*(j-1)):D*j, 1), 'fro'));
    end
    s = reshape(repmat(s./no, D, 1), [m*D, 1]);
    s = s/sqrt(D*ell); %the factor from equation (7)
end