using Pkg

Pkg.add("TRS")

include("./DynPrec.jl")
include("./functions/FunctionWrapper.jl")

# My Julia setup doesn't read my bashrc for some reason
# - you need to do whatever you need to do to give this
# environment variable to Julia

ENV["PREDUCER"] = "/Users/mattmenickelly/preducer"

global PROBLEM = "ROSENBR"

using TRS
using CUTEst
using NLPModels
using .DynPrec
using .Wrapper

function handle(x,prec)
    problem = PROBLEM
    f,g = Wrapper.wrapfun(x,problem,prec)
    return f,g
end

nlp = CUTEstModel(PROBLEM)
x0 = nlp.meta.x0
finalize(nlp)

prec_vec = [32;64]
epsilon = 1e-8
budget = 100

x,f = DynPrec.DynTR(x0,handle,prec_vec,epsilon,budget)
println(x)
println(f)
