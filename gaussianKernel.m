function K = gaussianKernel(x1,x2,sigma)

x1 = x1(:);
x2 = x2(:);

tmp = (x1 - x2).^2;
K = exp( - sum(tmp(:)) / (2*sigma.^2) );
    
end
