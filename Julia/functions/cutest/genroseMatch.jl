module Wrapper

export wrapfun

using CUTEst
using NLPModels

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

function wrapfun(x::AbstractVector,problem::String)
    nlp = CUTEstModel(problem, verbose=false)
    fx = obj(nlp, x)
    gx = grad(nlp, x)

    finalize(nlp)

    return convert(Float64,fx),convert(Array{Float64},gx)
end

y = ones(500)
A = genrose(y)
B = wrapfun(y,"GENROSE")
print(A)
print(B)