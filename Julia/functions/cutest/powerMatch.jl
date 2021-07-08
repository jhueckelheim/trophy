module Wrapper

export wrapfun

using CUTEst
using NLPModels

function power(x::AbstractVector)
  println("Julia port of CUTEST's GENROSE")
  grad = zeros(size(x))
  sum = 0
  for i = 1:length(x)
    term = i*x[i]^2
    sum = sum + term
  end
  for i = 1:length(x)
    grad[i] = 2*sum*2*i*x[i]
  end
  sum = sum^2
  return sum, grad
end

function wrapfun(x::AbstractVector,problem::String)
    nlp = CUTEstModel(problem, verbose=false)
    fx = obj(nlp, x)
    gx = grad(nlp, x)

    finalize(nlp)

    return convert(Float64,fx),convert(Array{Float64},gx)
end


y = ones(10000)
A = power(y)
B = wrapfun(y,"POWER")
print(A)
print(B)
end