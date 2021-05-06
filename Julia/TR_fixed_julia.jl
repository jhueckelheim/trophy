using LinearAlgebra

function [xvec,fvec,gvec] = TR_fixed(x0,fun,hessian,delta0,eps,budget)


## Check for hessian value 

 if !(hessian == 0 || hessian == 1)
     println("hessian value must be either 0 or 1.")
     return;
 end
 
X=x0[:];
n = size(X,1);                                                             # dimension of the problem


##### INITIALIZATION #####

n1 = 0.1;                                                                  # n1 denotes eta_1 such that 0 < eta_1 <= eta_2 .< 1
n2 = 0.75;                                                                 # n2 denotes eta_2
gamma1 = 0.5;                                                              # 0 < gamma_1 <= gamma_2 .< 1
gamma2 = 2
n0 = 0.04*n1;                                                              # n0 denotes eta_0 such that eta_0 .< 0.5*(eta_1)
kg= (0.5*(1-n2)-n0)-1;                                                     # kg denotes kappa_g such that kappa_g .< 0.5*(1-eta_2) -eta_0


## Computing f0 ##

xk =x0;

delta = delta0;
 
fvec = []; gvec = []; xvec = []; H = Diagonal(ones(n,n));
 
for k = 1:budget

    if k==1
        [f,g] = fun(xk)
    end
    
    if norm(g) <= eps/(1+kg)
        return                                                             # Terminate the program
    end
    
    gprev = g;
    
    ## Step calculation ##    
    # Try a Quasi-Newton approximation H = H + (y-H*s)*(y-H*s)/(y-H*s)'*y
    # SR1? BFGS? L-BFGS? L-SR1? 
    
#     [sk,val,~,~,~] = trust(g,zeros(n),delta);
    try
        [sk,val,~,~,~] = trust(g,H,delta);
    catch
        break;
    end
    
    #[sk,val,~,~,~] = trust(g,H,delta)
   
   # This is where precision_level should change. 
   # If the single/double/half choice changes here; then you need to 
   # recompute f;g at the new precision. 
   
    ## Evaluate the objective function ##
    
    [fplus,gplus] = fun(xk+sk);
    
    ## Acceptance of the trial point
    
    rhok = (f-fplus)./(-val);
    
    if rhok < n1
        xnew = xk;
    else
        xnew = xk + sk;
        f = fplus;
        g = gplus;
    end
    
    ## Radius update
    
    if rhok >= n2
        delta = gamma2*delta;     
    elseif rhok <= n1
        delta = gamma1*delta;
    end
    
   ## Updating Hessian using SR-1 update
    
    y = gplus-gprev;
    
    if hessian == 1
        grad_pred = y-H.*sk;
        dp = (grad_pred)'.*sk;
        update = ((grad_pred).*(grad_pred)'./dp)
        if abs(dp) >= 1e-8*norm(grad_pred)*norm(sk) && (!any(any(isnan(update))) && !any(any(isinf(update))))
            H = H + update;
        end     
    elseif hessian == 0
        H = zeros(n,n);
    end
    
    
    fvec = cat(1,fvec,f); gvec = cat(1,gvec,norm(g));
    xk = xnew; xvec = cat(2,xvec,xk);
    printf("k = %d, f = %.8f, norm(g) = %.8f \n",k,f,norm(g))
    if delta <= 1e-16                                                       # there"s no way you"re making progress at this point
        break;
    end
end


end