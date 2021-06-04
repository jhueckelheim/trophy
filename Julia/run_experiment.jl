using Pkg

Pkg.add(path="/Users/clancy/TRS")

include("./DynPrec.jl")
include("./functions/FunctionWrapper.jl")

# My Julia setup doesn't read my bashrc for some reason
# - you need to do whatever you need to do to give this
# environment variable to Julia

ENV["PREDUCER"] = "/Users/clancy/repos/preducer" #mattmenickelly/preducer"

using TRS
using CUTEst
using NLPModels
using .DynPrec
using .Wrapper
using MAT

problems = CUTEst.select(min_var=2,max_var=10,contype="unc")
println(problems)
prec_vec = [32;64]
#prec_vec = [32];
epsilon = 1e-8
# 14, 82, 85 are all bad
for prob_no = 1:length(problems)
    if prob_no != 14 && prob_no != 82 && prob_no != 85
        println(problems[prob_no])
        global PROBLEM = problems[prob_no]
        function handle(x,prec)
            problem = PROBLEM

            f,g = Wrapper.wrapfun(x,problem,prec)
            return f,g
        end
        nlp = CUTEstModel(PROBLEM)
        x0 = nlp.meta.x0
        n = nlp.meta.nvar
        budget = 50*(n+1)
        finalize(nlp)
        prechist, xhist = DynPrec.DynTR(x0,handle,prec_vec,epsilon,budget)
        println(x0)
        print(xhist)
        #println(xhist[1])

        println("completed", prob_no, "/", length(problems))
        sleep(1)
        readline()
        # save benchmarking files:
        savename = string("prob", prob_no, ".mat")
        #savename = string("dprob", prob_no, ".mat")
        file = matopen(savename, "w")
        write(file, "xhist", xhist)
        write(file, "prechist", prechist)
        close(file)
    end
end
# x,f = DynPrec.DynTR(x0,handle,prec_vec,epsilon,budget)
# println(x)
# println(f)
