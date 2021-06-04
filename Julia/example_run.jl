using Pkg

Pkg.add(path="/Users/clancy/TRS")

include("./DynPrec.jl")
include("./functions/FunctionWrapper.jl")

# My Julia setup doesn't read my bashrc for some reason
# - you need to do whatever you need to do to give this
# environment variable to Julia

ENV["PREDUCER"] = "/Users/clancy/repos/preducer" #/Users/mattmenickelly/preducer

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

nlp = CUTEstModel(PROBLEM)      # this takes in a problem string and gets info t
x0 = nlp.meta.x0                # initializion
finalize(nlp)                   # kill problem already?

prec_vec = [32;64]      # set different precision levels.
epsilon = 1e-8          # set tolerance
budget = 100            # number of iterations to do

x,f = DynPrec.DynTR(x0,handle,prec_vec,epsilon,budget)
println(x)
println(f)
