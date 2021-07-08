module Wrapper

export wrapfun

using CUTEst
using NLPModels

function fletchcr(x::AbstractVector)
  println("Julia port of CUTEST's FLETCHCR")
  grad = zeros(size(x))
  sum = 0
  for i = 1:(length(x)-1)
    term1 = -x[i]^2 + x[i+1]
    term2 = -x[i]+1
    sum = sum + 100*term1^2 + term2^2
    grad[i] = grad[i] + 2*100*term1*-2*x[i] + 2*term2*-1
    grad[i+1] = grad[i+1] + 2*100*term1
  end
  return sum, grad
end

function wrapfun(x::AbstractVector,problem::String)
    nlp = CUTEstModel(problem, verbose=false)
    fx = obj(nlp, x)
    gx = grad(nlp, x)

    finalize(nlp)

    return convert(Float64,fx),convert(Array{Float64},gx)
end

y = ones(1000)
A = fletchcr(y)
B = wrapfun(y,"FLETCHCR")
print(A)
print(" ")
print(B)
print(" ")
end