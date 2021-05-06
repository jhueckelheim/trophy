include("./functions/FunctionWrapper.jl")

ENV["PREDUCER"] = "/Users/mattmenickelly/preducer"

using CUTEst
using NLPModels
using .Wrapper
using MAT
using LinearAlgebra

problems = CUTEst.select(min_var=1,max_var=10,contype="unc")
for prob_no = 1:length(problems)
    if prob_no != 14 && prob_no != 82 && prob_no != 85
        global PROBLEM = problems[prob_no]
        function handle(x,prec)
            problem = PROBLEM
            f,g = Wrapper.wrapfun(x,problem,prec)
            return f,g
        end
        savename = string("prob", prob_no, ".mat")
        processedsavename = string("fprob",prob_no,".mat")
        if isfile(savename) && ~isfile(processedsavename)
            file = matopen(savename)
            prechist = read(file,"prechist")
            xhist = read(file,"xhist")
            close(file)
            fevals = length(prechist)
            xhist = xhist[2:min(size(xhist)[1],fevals),:]
            num_fevals = size(xhist)[1]
            fvals = zeros(Float64,1,num_fevals)
            normgvals = zeros(Float64,1,num_fevals)
            for i = 1:num_fevals
                f,g = handle(xhist[i,:],64)
                fvals[i] = f
                normgvals[i] = norm(g)
            end
            fsavename = string("f",savename)
            file = matopen(fsavename, "w")
            write(file, "fvals", fvals)
            write(file, "normgvals", normgvals)
            close(file)
        end
        println("completed", prob_no, "/", length(problems))
    end
end
