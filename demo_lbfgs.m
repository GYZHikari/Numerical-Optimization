% demo for L-BFGS
% Guangyu Zhong@2014/06/09
% Guangyuzhonghikari@gmail.com
clear; clc;
syms x1 x2;
k = 1;
f=(0.5*x1^2+0.5*x2^2)+...
    3*(1-x1)^2*exp(-(x1^2) - (x2+1)^2)- ...
    10*(x1/5 - x1^3 - x2^5)*exp(-x1^2-x2^2)- ...
    1/3*exp(-(x1+1)^2 - x2^2);
df = jacobian(f);
hessian = jacobian(df);


epsilon = 5;
mm = 10;
s = zeros(2, mm);
y = zeros(2, mm);
x = [0,0];
mm = 10;
ff(k) = double(subs(f,[x1, x2], x(k,:))');
max_num = 20;
linermode = 'wolfe';
for ii = 1:max_num
    if (epsilon<0.001)
        break;
    end
   
    r = (s(:,k)'*y(:,k))/(y(:,k)'*y(:,k));
    if k == 1
        H0 = eye(2);
    else
        H0 = eye(2)*r;
    end
    g = double(subs(df,[x1, x2], x)');
    p = -lbfgs(g,s(:,1:k),y(:,1:k),H0);
    
    switch lower(linermode)
        case 'armijo'
            alpha = 1;rou = 0.1;c =  0.8;
            f_k = double(subs(f, [x1, x2], x)');
            
            for nn = 1:100
                f_k1= double(subs(f, [x1, x2], x+alpha*p'));
                if ((f_k1 - f_k)<c*alpha*g'*p)
                    break;
                else
                    alpha = rou*alpha;
                end
            end
        case 'wolfe'
            %% wolfe
            rho = 0.25; sigma = 0.75;
            alpha = 1; a = 0; b = Inf;
            while (1)
                if ~(double(subs(f, [x1, x2], x+alpha*p'))<=double(subs(f, [x1, x2], x)+rho*alpha*double(subs(df,[x1, x2], x)')*p'))
                    b = alpha;
                    alpha = (alpha+a)/2;
                    continue;
                end
                if ~(double(subs(df,[x1, x2], x+alpha*p')*p >= sigma*double(subs(df,[x1, x2], x+alpha*p')*p)))
                    a = alpha;
                    alpha = min([2*alpha, (b+alpha)/2]);
                    continue;
                end
                break;
            end
    end
    
    k = k + 1;
    xnew = x + alpha*p';
    if k<=mm
        s(:,k) = alpha*p';
        y(:,k) =  double(subs(df,[x1, x2], xnew)') -  double(subs(df,[x1, x2], x)');
    else
        s = [s(:,2:mm) alpha*p];
        y = [y(:,2:mm) double(subs(df,[x1, x2], xnew)') -  double(subs(df,[x1, x2], x)')];
        k = mm;
    end
    ff = double(subs(f,[x1, x2], xnew))
    x = xnew;
end
    x
    X1 = 0:1:10;
    X2 = 0:1:10;
    for jj = 1:length(X1)
        for uu = 1:length(X2)
            U(jj,uu) = double(subs(f,[x1, x2], [X1(jj),X2(uu)]));
        end
    end
    figure;
    surf(X1, X2, U);
    xlabel('x1'); ylabel('x2');
    hold on;
    plot3(x(1),x(2),ff,'o','color','r');
    
    figure;
    contour(X1, X2, U, logspace(-2, 3, 20));hold on;
    
    plot(x(1), x(2), 'rx', 'MarkerSize', 10, 'LineWidth', 2);

