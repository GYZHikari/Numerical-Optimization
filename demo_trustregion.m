% demo for trust-region
% Guangyu Zhong @ 2014/05/05
% guangyuzhonghikari@gmail.com
clear; clc; close all
syms x1 x2;


f=(0.5*x1^2+0.5*x2^2)+...
    3*(1-x1)^2*exp(-(x1^2) - (x2+1)^2)- ...
    10*(x1/5 - x1^3 - x2^5)*exp(-x1^2-x2^2)- ...
    1/3*exp(-(x1+1)^2 - x2^2);
% f = x1^4 + x1^2 + x2^2 - 4*x2 + 5;
df = jacobian(f);
hessen = jacobian(df);
tau = 2;
% initial
delta_hat =5; delta = 3; ita = 0.01;
kmax = 100;
x = [0,0];
epsilon = 0.001;
sigma = 2;
k =0;
x_mark = x;

% mode = 'dog-leg';
mode = 'optimal';
while(k<kmax)
    gk = double(subs(df, [x1, x2], x))';
    if sqrt(sum(gk.^2))<epsilon
        break;
    end
    Bk = subs(hessen, [x1, x2], x);
    Bk = double(Bk);

    ff = double(subs(f, [x1, x2], x));
    p = solve_semi(gk, Bk, delta, mode,tau,epsilon);
    
    ff_1 = double(subs(f, [x1, x2], (x'+p)'));
    
    mk= ff + double(subs(df, [x1, x2], x))*p + 0.5*p'*Bk*p;
    rou = (ff - ff_1)/(ff - mk);
    
    if rou < 0.25
        delta = 0.25*sqrt(sum(p.*p));
    else
        if rou >0.75&&sqrt(sum(p.*p)) == delta
            delta = min(2*delta, delta_hat);
        end
    end
    if rou > ita
        x = (x' + p)';
    end
    k = k+1;
    x_mark =[x_mark; x];
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
plot(x_mark(:,1), x_mark(:,2), 'rx', 'MarkerSize', 10, 'LineWidth', 2);
x
