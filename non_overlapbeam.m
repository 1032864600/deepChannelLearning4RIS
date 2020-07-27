function index =  non_overlapbeam(N,L,K)
x = randperm(N);
y = x(1:L*K);
index = reshape(y,L,K);