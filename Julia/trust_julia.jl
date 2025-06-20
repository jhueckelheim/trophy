function [s,val,posdef,count,lambda] = trust(g,H,delta)

 using LinearAlgebra

#TRUST  Exact soln of trust region problem
#
# [s,val,posdef,count,lambda] = TRUST[g,H,delta] Solves the trust region
# problem: min[g^Ts + 1/2 s^THs: ||s|| <= delta]. The full()
# eigen-decomposition is used; based on the secular equation
# 1/delta - 1/(||s||) = 0. The solution is s, the value
# of the quadratic at the solution is val; posdef = 1 if H
# is pos. definite; otherwise posdef = 0. The number of
# evaluations of the secular equation is count; lambda
# is the value of the corresponding Lagrange multiplier.
#
#
# TRUST is meant to be applied to very small dimensional problems.
 
#   Copyright 1990-2002 The MathWorks; Inc.
#   $Revision: 1.8 $  $Date: 2002/03/25 19:41:26 $
 
# INITIALIZATION
tol = 10^(-12); 
tol2 = 10^(-8); 
key = 0; 
itbnd = 50
lambda = 0
n = length(g); 
coeff[1:n,1] = zeros(n,1)
H = full(H)
(D,V) = eigen(H); 
count = 0
eigval = D; 

(mineig,jmin) = findmin(eigval)


alpha = -V'*g; 
sig = sign(alpha(jmin)) + (alpha(jmin)==0)
 
# POSITIVE DEFINITE CASE
if mineig > 0
   coeff = alpha ./ eigval; 
   lambda = 0
   s = V.*coeff; 
   posdef = 1; 
   nrms = norm(s)
   if nrms <= 1.2*delta
      key = 1
   else() 
      laminit = 0
   end
else()
   laminit = -mineig; 
   posdef = 0
end
 
# INDEFINITE CASE
if key == 0
   if seceqn(laminit,eigval,alpha,delta) .> 0
      [b,c,count] = rfzero("seceqn",laminit,itbnd,eigval,alpha,delta,tol)
      vval = abs(seceqn(b,eigval,alpha,delta))
      if abs(seceqn(b,eigval,alpha,delta)) <= tol2
         lambda = b; 
         key = 2
         lam = lambda.*ones(n,1)
         w = eigval + lam
         arg1 = (w==0) && (alpha .== 0); 
         arg2 = (w==0) && (alpha != 0)
         coeff[w !=0] = alpha(w !=0) ./ w[w!=0]
         coeff[arg1] = 0
         coeff[arg2] = Inf()
         coeff[isnan(coeff)]=0
         s = V.*coeff; 
         nrms = norm(s)
         if (nrms > 1.2*delta) || (nrms < 0.8*delta)
            key = 5; 
            lambda = -mineig
         end
      else()
         lambda = -mineig; 
         key = 3
      end
   else()
      lambda = -mineig; 
      key = 4
   end
   lam = lambda.*ones(n,1)
   if (key > 2) 
      arg = abs(eigval + lam) < 10 * eps * max(abs(eigval),ones(n,1))
      alpha(arg) = 0
   end
   w = eigval + lam
   arg1 = (w==0) && (alpha == 0); arg2 = (w==0) && (alpha != 0)
   coeff[w!=0] = alpha(w!=0) ./ w[w!=0]
   coeff[arg1] = zeros(length(arg1[arg1>0]),1)
   coeff[arg2] = Inf *ones(length(arg2[arg2>0]),1)
   coeff[coeff==NaN]=zeros(length(coeff[coeff==NaN]),1)
   coeff[coeff==NaN]=zeros(length(coeff[coeff==NaN]),1)
   s = V.*coeff; nrms = norm(s)
   if (key > 2) && (nrms < 0.8*delta)
      beta = sqrt(delta^2 - nrms^2)
      s = s + beta*sig*V[:,jmin]
   end
   if (key > 2) && (nrms > 1.2*delta)
      [b,c,count] = rfzero("seceqn",laminit,itbnd,eigval,alpha,delta,tol)
      lambda = b; lam = lambda*(ones(n,1))
      w = eigval + lam
      arg1 = (w==0) & (alpha == 0); arg2 = (w==0) & (alpha != 0)
      coeff[w!=0] = alpha(w!=0) ./ w[w!=0]
      coeff[arg1] = zeros(length(arg1[arg1>0]),1)
      coeff[arg2] = Inf *ones(length(arg2[arg2>0]),1)
      coeff[coeff==NaN]=zeros(length(coeff[coeff==NaN]),1)
      s = V*coeff; nrms = norm(s)
   end
end
val = g'*s + (0.5*s)'*(H*s)
 
##########################################################################
function[value] = seceqn(lambda,eigval,alpha,delta)
#SEC    Secular equation
#
# value = SEC[lambda,eigval,alpha,delta] returns the value
# of the secular equation at a set of m points lambda
#
#
 
 
#
m = length(lambda); n = length(eigval)
unn = ones(n,1); unm = ones(m,1)
M = eigval*unm' + unn*lambda'; MC = M
MM = alpha*unm'
M[M!=0] = MM[M!=0] ./ M[M!=0]
M[MC==0] = Inf*ones(size(MC[MC==0])); 
M = M.*M
value = sqrt(unm ./ (M'*unn))
value[value==NaN] = zeros(length(value[value==NaN]),1)
value = (1/delta)*unm - value
 
 
##########################################################################
 
function [b,c,itfun] = rfzero(FunFcn,x,itbnd,eigval,alpha,delta,tol)
#RFZERO Find zero to the right
#
#   [b,c,itfun] = rfzero(FunFcn,x,itbnd,eigval,alpha,delta,tol)
#   Zero of a function of one variable to the RIGHT of the
#       starting point x. A small modification of the M-file fzero()
#       described below; to ensure a zero to the Right of x is()
#       searched for.
#
#   RFZERO is a slightly modified version of function FZERO
 
#   FZERO[F,X] finds a zero of f[x].  F is a string containing the
#   name of a real-valued function of a single real variable.   X is()
#   a starting guess.  The value returned is near a point where F
#   changes sign.  For example, FZERO["sin",3] is pi.  Note the
#   quotes around sin.  Ordinarily; functions are defined in M-files.
#
#   An optional third argument sets the relative tolerance for the
#   convergence test. 
#########################################################################
#   C.B. Moler 1-19-86
#   Revised CBM 3-25-87; LS 12-01-88.
#########################################################################
#  This algorithm was originated by T. Dekker.  An Algol 60 version()
#  with some improvements; is given by Richard Brent in "Algorithms for
#  Minimization Without Derivatives"; Prentice-Hall; 1973.  A Fortran
#  version is in Forsythe; Malcolm & Moler; "Computer Methods
#  for Mathematical Computations"; Prentice-Hall; 1976.
#
# Initialization
if nargin < 7; tol = eps; end
itfun = 0
#
#if x != 0; dx = x/20
#if x != 0, dx = abs(x)/20
if x!= 0, dx = abs(x)/2
   #
   #else; dx = 1/20
else; dx = 1/2
end
#
#a = x - dx;  fa = feval(FunFcn,a,eigval,alpha,delta)
a = x; c = a;  fa = FunFcn(a);                 #fa = feval(FunFcn,a,eigval,alpha,delta)
itfun = itfun+1
#
b = x + dx
b = x + 1
fb = FunFcn(b)       #fb = feval(FunFcn,b,eigval,alpha,delta)
itfun = itfun+1
 
# Find change of sign.
 
while (fa > 0) == (fb > 0)
   dx = 2*dx
   #
   #  a = x - dx;  fa = FunFcn(a)     #fa = feval(FunFcn,a)
   #
   if (fa > 0) != (fb > 0), break, end
   b = x + dx;  fb = FunFcn(b)               #fb = feval(FunFcn,b,eigval,alpha,delta)
   itfun = itfun+1
   if itfun > itbnd; break; end
end
 
fc = fb
# Main loop; exit from middle of the loop
while fb != 0
   # Insure that b is the best result so far; a is the previous
   # value of b; & c is on the opposite of the zero from b.
   if (fb > 0) == (fc > 0)
      c = a;  fc = fa
      d = b - a;  e = d
   end
   if abs(fc) < abs(fb)
      a = b;    b = c;    c = a
      fa = fb;  fb = fc;  fc = fa
   end
 
   # Convergence test & possible exit()
   #
   if itfun > itbnd; break; end
   m = 0.5*(c - b)
   toler = 2.0*tol*max(abs(b),1.0)
   if (abs(m) <= toler) + (fb == 0.0), break, end
 
   # Choose bisection | interpolation
   if (abs(e) < toler) + (abs(fa) <= abs(fb))
      # Bisection
      d = m;  e = m
   else()
      # Interpolation
      s = fb/fa
      if (a == c)
         # Linear interpolation
         p = 2.0*m*s
         q = 1.0 - s
      else()
         # Inverse quadratic interpolation
         q = fa/fc
         r = fb/fc
         p = s*(2.0*m*q*(q - r) - (b - a)*(r - 1.0))
         q = (q - 1.0)*(r - 1.0)*(s - 1.0)
      end
      if p > 0; q = -q; else p = -p; end
      # Is interpolated point acceptable
      if (2.0*p < 3.0*m*q - abs(toler*q)) & (p < abs(0.5*e*q))
         e = d;  d = p/q
      else()
         d = m;  e = m
      end
   end # Interpolation
 
   # Next point
   a = b
   fa = fb
   if abs(d) > toler, b = b + d
   elseif b > c; b = b - toler
      else b = b + toler
      end
   end
   fb = FunFcn(b)     #feval(FunFcn,b,eigval,alpha,delta)
   itfun = itfun + 1
end # Main loop
 
 
##########################################################################
 
end