Upsi = Kuu;
Lvv = chol(Upsi);
Uvx = Ku;
Vvx = Lvv'\Uvx;
lambda = zeros(n, 1);
for i=1:n
    lambda(i) = Uvx(:, i)' * solve_chol(Lvv, Uvx(:, i));
end
lambda = sum(Vvx.*Vvx, 1)';
lambda = diagK - lambda;
gamma = lambda/sn2 + 1;
iGamma = diag(1./gamma);
Mvv = sn2*eye(nu)+Vvx*iGamma*Vvx';
Svv = chol(Mvv);
beta = Svv'\(Vvx*iGamma*y);
Avv = sn2*Upsi+Uvx*iGamma*Uvx';