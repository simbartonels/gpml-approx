function [ J, lambda ] = initHSM( m, D, L )
%INITHSM Summary of this function goes here
%   Detailed explanation goes here

    lambda = zeros(m^D, 1);
    J = getIndexMatrix(D, m);
    for k = 1:m^D
        lambda(k) = pi^2*sum((J(:, k)'./(2*L)).^2);
    end
end

function J = getIndexMatrix(D, M)
    J = ones(D, M^D);
    for d = 1:(D-1)
        J(d, :) = repmat(reshape(repmat((1:M)', 1, M.^(D-d))', [M.^(D-d+1), 1]), M.^(d-1), 1);
    end
    J(D, :) = repmat((1:M)', M.^(D-1), 1);
end