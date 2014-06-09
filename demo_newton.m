% demo for newton method
% Guangyu zhong @ 2014/04/20
% Guangyuzhonghikari@gmail.com
clear;
clc;
close all;
syms x1 x2;

f=(0.5*x1^2+0.5*x2^2)+...
    3*(1-x1)^2*exp(-(x1^2) - (x2+1)^2)- ...
    10*(x1/5 - x1^3 - x2^5)*exp(-x1^2-x2^2)- ...
    1/3*exp(-(x1+1)^2 - x2^2);
df = jacobian(f);
hessen = jacobian(df);
x = [0, 0];
maxk = 200;
k = 1;
epsilon = 0.1;
sigma =0.01;
linear_mode = 'armijo';
while(k<maxk)
    delta_f = double(subs(df,[x1, x2], x)');
    if double(norm(delta_f))< epsilon
        break;
    end
    hessen = subs(hessen,[x1, x2], x);
    hessen = double(hessen);
    [L,p]=chol(hessen);
    
    if p~=0
        [u,lambda] = eig(hessen);
        [index] = find(diag(lambda)<sigma);
        
        for ii = 1:numel(index)
            lambda(index(ii), index(ii)) = sigma - lambda(index(ii), index(ii));
        end
        hessen = u*lambda*inv(u);
    end
    dk = -hessen\delta_f;
    %% liner search
    switch lower(linear_mode)
        case 'armijo'
            alpha = 1;rou = 0.1;c =  0.8;
            f_k = double(subs(f, [x1, x2], x)');
            
            for nn = 1:100
                f_k1= double(subs(f, [x1, x2], x+alpha*dk'));
                if ((f_k1 - f_k)<c*alpha*delta_f'*dk)
                    break;
                else
                    alpha = rou*alpha;
                end
            end
            x = x + alpha*dk';
        case 'wolfe'
            %% wolfe
            rho = 0.25; sigma = 0.75;
            alpha = 1; a = 0; b = Inf;
            while (1)
                if ~(double(subs(f, [x1, x2], x+alpha*dk'))<=double(subs(f, [x1, x2], x)+rho*alpha*double(subs(df,[x1, x2], x)')*dk'))
                    b = alpha;
                    alpha = (alpha+a)/2;
                    continue;
                end
                if ~(double(subs(df,[x1, x2], x+alpha*dk')*dk >= sigma*double(subs(df,[x1, x2], x+alpha*dk')*dk)))
                    a = alpha;
                    alpha = min([2*alpha, (b+alpha)/2]);
                    continue;
                end
                break;
            end
            x = x+alpha*dk';
    end   
    k = k+1;
end 
    minf = double(subs(f,[x1, x2], x));
    
    len = abs(fix(2* max(x)));
    X1 = 0:1:max(len, 10);
    X2 = 0:1:max(len, 10);
    for jj = 1:length(X1)
        for uu = 1:length(X2)
            U(jj,uu) = double(subs(f,[x1, x2], [X1(jj),X2(uu)]));
        end
    end
    figure;
    surf(X1, X2, U);
    xlabel('x1'); ylabel('x2');
    hold on;
    plot3(x(1),x(2),minf,'o','color','r');
    
    figure;
    contour(X1, X2, U, logspace(-2, 3, 20));hold on;   
    plot(x(1), x(2), 'rx', 'MarkerSize', 10, 'LineWidth', 2);
