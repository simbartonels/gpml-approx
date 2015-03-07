function SMSGPR_example()
%TODO: Maybe use step function and place inducing points with short length
%scales near the step. And others with longer length scales further away.
%OR COMPARE TO STANDARD FIC.

%seed = 22124;
seed = randi(32000)
rng(seed);
indNoise = -Inf;
D = 1;
n = 20;
logell = -2;

%V = [0.15; 1; 1.75; 2.5];
V = [0.15; 1; 2.5];
logsigma = [2*logell; logell/2; logell/2];
%V = [0.2; 1.5];
%logsigma = [2*logell; 1];

logsigma = [logsigma; logsigma];
V = [-V; V];
M = size(V, 1);
X = linspace(-3, 3, n)';
%X = randn(n+2, D);
y = sign(X); %randn(n, 1);
%p = randperm(n);
%V = X(p(1:M), :);
%V = randn(M, D);
%V = X(n+2-M+1:n+2, :);
hyp.lik = -20;
%logell = randn([D, 1]);
%logsigma = randn([M, 1]);
sf2 = 0; %randn(1);
hyp.cov = [logell; reshape(logsigma, [M*D, 1]); ...
        reshape(V, [M*D, 1]); sf2];
covfunc = {@covSM, M};
infMethod = @infFITC;

%covfunc = {@covFITC, {@covSEard}, V};
%infMethod = @infFITC;
%hyp.cov = zeros([D+1, 1]);

%hyp = minimize(hyp, @gp, -100, infMethod, [], covfunc, @likGauss, X, y);

xs = linspace(min(X), max(X), 50)';   
[mF, s2F] = gp(hyp, infMethod, [], covfunc, @likGauss, X, y, xs);
[mV, s2V] = gp(hyp, infMethod, [], covfunc, @likGauss, X, y, V);

%figure('units','normalized','position',[.1 .1 .6 .4])
figure('units','centimeters','position',[10 10 15 5])
hold on;
axis([min(X) max(X) -3 3]);
%fill([xs',fliplr(xs')],[(mF+s2F)', fliplr((mF-s2F)')], 'r', 'FaceAlpha', 0.25);
%plot(xs, mF, '-', 'Color', 'r');
plot(X, y, '+', 'Color', 'k');
%plot(V, 0, 'x', 'Color', 'b');
end

