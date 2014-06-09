function sk = solve_semi(gk,  Bk, delta, mode,tau ,epsilon)
% gk: gradient 
% Bk: Hessian
% Guangyu Zhong @ 2014/05/05
% guangyuzhonghikari@gmail.com
if nargin < 4
    mode = 'dog-leg';
end


[L,q]=chol(Bk);
    
    if q~=0
        [u,lambda] = eig(Bk);
        tt = max(0, 1 - lambda(1));
        Bk = Bk + tt*eye(2);
    end

switch lower(mode)
    case 'dog-leg'
        alpha = (gk'*gk)/(gk'*Bk*gk);
        skc = -alpha*gk;
        skn = -Bk\gk;
        if sqrt(sum(skc.^2))>=delta
            sk = -(delta*gk) / sqrt(sum(gk.^2));
        else
            if tau>=0&&tau<=1
                sk = tau*skc;
            end
            if  tau>=1&&tau<=2
                sk = skc + (tau - 1)*(skn - skc);
            end
        end
    case 'optimal'
        if sqrt(sum((-gk\Bk).^2))<delta
            sk = -Bk\gk;
        else
            sk = trustq(gk,Bk,delta);
        end

end