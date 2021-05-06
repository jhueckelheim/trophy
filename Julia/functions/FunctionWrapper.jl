module Wrapper

export wrapfun

using CUTEst
using NLPModels

function extrose(x::AbstractVector)
  println("Extended Rosenbrock function as found in")
  println("Neculai Andrei, 'An Unconstrained Optimization Test Functions Collection'")
  grad = zeros(size(x))
  sum = 0
  for i = 1:length(x)÷2
    term = (x[2*i] - x[2*i-1]^2)
    sum = sum + 100*term^2 + (1 - x[2*i-1])^2
    grad[2*i-1] = grad[2*i-1] - 2 * (1-x[2*i-1]) - 400*x[2*i-1]*term
    grad[2*i] = grad[2*i] + 200*term
  end
  return sum, grad
end

function genrose(x::AbstractVector)
  println("Julia port of CUTEST's GENROSE")
  grad = zeros(size(x))
  term = (1 - x[1])
  grad[1] = 2 * term
  sum = 1 - term^2
  term = (x[size(x,1)] - 1)
  grad[size(x,1)] = 2 * term
  sum = sum + term^2
  for i = 1:length(x)-1
    term = (x[i+1] - x[i]^2)
    sum = sum + 100*term^2 + (1 - x[i])^2
    grad[i] = grad[i] - 2 * (1-x[i]) - 400*x[i]*term
    grad[i+1] = grad[i+1] + 200*term
  end
  return sum, grad
end

function extrosnb(x::AbstractVector)
  println("Julia port of CUTEST's EXTROSNB")
  grad = zeros(size(x))
  term = (1 - x[1])
  sum = term^2
  grad[1] = -2*term
  for i = 1:length(x)-1
    term = (x[i+1] - x[i]^2)
    sum = sum + 100*term^2
    grad[i+1] = grad[i+1] + 200*term
    grad[i] = grad[i] - 400*x[i]*term
  end
  return sum, grad
end

# Generalized Rosenbrock function. Computes both function and gradient.

function genrosen(x::AbstractVector)
   #println("Problem 1")
   item1 = [10 * (x[i] - x[i - 1] * x[i - 1]) for i=2:length(x)]
   item2 = x[2:length(x)] - ones(length(x)-1)
   return item1'*item1 + item2'*item2, -40 * [item1;0] .* x + [0; 20*item1 + 2*item2]
end

# Slightly different implementation of the generalized Rosenbrock function. Computes both function and gradient.

function genrosen2(x::AbstractVector)
   #println("Problem 2")
   temp1 = 10*(x[2:length(x)] - x[1:length(x)-1].*x[1:length(x)-1])
   temp2 = x[2:length(x)] - ones(length(x)-1)
   return temp1'*temp1 + temp2'*temp2, -40 * [temp1;0] .* x + [0; 20*temp1 + 2*temp2]
end

#= Function wrapper to compute a function and gradient at a given level
   of precision.

   parameters:
   x - point at which to evaluate the function
   problem - problem identifier
   precision - # of mantissa bits for both function and gradient
               (uses as many exponent bits as necessary)
               (53 should be double precision, 24 single, 11 half, 8 bfloat16)
   return values:
   f - function value (in double precision)
   g - gradient values (double precision array)
=#
function wrapfun(x::AbstractVector,problem::String,precision::Integer)
    if precision == 32
        nlp = CUTEstModel(problem, verbose=false, "-preduce")
    else
        nlp = CUTEstModel(problem, verbose=false)
    end
    fx = obj(nlp, x)
    gx = grad(nlp, x)

    finalize(nlp)

    return convert(Float64,fx),convert(Array{Float64},gx)
end


function wrapfun(x::AbstractVector,problem::Integer,precision::Integer)
   setprecision(precision)
   bx = convert(Array{BigFloat},x)
   if problem == 1
      f,g = genrosen(bx)
   elseif problem == 2
      f,g = genrosen2(bx)
   elseif problem == 3
      f,g = extrose(bx)
   elseif problem == 4
      f,g = genrose(bx)
   elseif problem == 5
      f,g = extrosnb(bx)
   end
   return convert(Float64,f),convert(Array{Float64},g)
end

#= Function wrapper to compute a function and gradient at a given level
   of precision.

   parameters:
   x - point at which to evaluate the function
   problem - problem identifier
   fprecision - # of mantissa bits for function
   gprecision - # of mantissa bits for gradient

   return values:
   f - function value (in double precision)
   g - gradient values (double precision array)
=#

function wrapfun(x::AbstractVector,problem::Integer,fprecision::Integer,gprecision::Integer)
   if fprecision == gprecision
      return wrapfun(x,fprecision)
   end
   setprecision(fprecision)
   bx = convert(Array{BigFloat},x)
   f, = genrosen(bx)
   setprecision(gprecision)
   bx = convert(Array{BigFloat},x)
   dummy,g = genrosen(bx)
   return convert(Float64,f),convert(Array{Float64},g)
end

end
