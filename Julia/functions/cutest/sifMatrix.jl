using Base: File, Integer
using LinearAlgebra

#BENCHMARK USEFUL SECTION LINES
function counter(problem::String,homeDir)
    lineVar = 0
    line = 0
    lineGroup = 0
    lineGroEnd = 0
    lineEleEnd = 0
    lineEnd1 = 0
    lineGroUse = 0
    lineEleUse = 0
    lineEleType = 0
    lineGroType = 0
    lineBoundz = 0
    lineConstz = 0
    lineObjBound = 0
    lineQuads = 0
    for k = 1:length(homeDir[:,1])
            #println("$line . $s")
            line+= 1
            if homeDir[k,1] == "VARIABLES" || homeDir[k,1] == "COLUMNS" 
                lineVar = k
            end
            if homeDir[k,1] == "GROUPS" || homeDir[k,1] == "ROWS" || homeDir[k,1] == "CONSTRAINTS" 
                lineGroup = k
            end
            if homeDir[k,1] == "GROUP TYPE"
                lineGroType = k
            end
            if homeDir[k,1] == "ELEMENT TYPE"
                lineEleType = k
            end
            if homeDir[k,1] == "ELEMENT USES"
                lineEleUse = k
            end
            if homeDir[k,1] == "GROUP USES"
                lineGroUse = k
            end
            if homeDir[k,1] == "ENDATA"
                lineEnd1 = k
            end
            a = homeDir[k,1]
            if length(a)>=10 && a[1:10] == "ELEMENTS  "
                lineEleEnd = k
            end
            if length(a)>=8 && a[1:8] == "GROUPS  "
                lineGroEnd = k
            end
            if homeDir[k,1] == "BOUNDS"
                lineBoundz = k
            end
            if homeDir[k,1] == "CONSTANTS" || homeDir[k,1] == "RHS" || homeDir[k,1] == "RHS'"  
                lineConstz = k
            end
            if homeDir[k,1] == "OBJECT BOUND"
                lineObjBound = k
            end
            if homeDir[k,1] == "QUADRATIC" || homeDir[k,1] == "HESSIAN" || homeDir[k,1] == "QUADS" || homeDir[k,1] == "QUADOBJ" || homeDir[k,1] == "QSECTION"
                lineQuads = k
            end
        
    
    end
    return line, lineVar, lineGroup, lineBoundz, lineConstz, lineEleType, lineEleUse, lineGroType, lineGroUse, lineObjBound, lineEleEnd, lineGroEnd, lineEnd1, lineQuads
end

#CONSTRUCT MATRIX OF SIF FILE
function house(problem::String)
    homeDir = Array{Union{Missing, String}}(missing, 1, 6)
    open(problem) do f
        line=0
        while ! eof(f)
            s = readline(f)
            str = s
            line += 1
            if length(str)>=1 && str[1] == '*'
                str=""
            end
            if length(str)>=4 && str[1:4] == "NAME"
                str=""
            end
            b=split(str,"\$")
            if length(b) > 1
                str = b[1]
            end
            home=Array{Union{Missing, String}}(missing, 1, 6)
            #NAME CARDS
            if length(str) >= 1 && str[1] != ' '
                home[1] = str
                str=""
            end
            #FIELD1
            if length(str) >= 1 && str[1] == ' '
                if length(str) >= 3
                    a = str[1:3]
                    a = replace(a," "=>"")
                    home[1] = a
                end
            end
            #FIELD2
            if length(str) > 3 && length(str) <= 13 
                a = str[4:length(str)]
                a = replace(a," "=>"")
                home[2] = a
            end
            if length(str) > 13 
                a = str[4:13]
                a = replace(a," "=>"")
                home[2] = a
            end
            #FIELD3
            if length(str) > 13 && length(str) <= 23 
                a = str[14:length(str)]
                a = replace(a," "=>"")
            home[3] = a
            end
            if length(str) > 23 
                a = str[14:23]
                a = replace(a," "=>"")
                home[3] = a
            end
            #FIELD4
            if length(str) > 23 && length(str) <= 35 
                a = str[24:length(str)]
                a = replace(a," "=>"")
                home[4] = a
            end
            if length(str) > 35 
                a = str[24:35]
                a = replace(a," "=>"")
                home[4] = a
            end
            #FIELD5
            if length(str) > 35 && length(str) <= 45 
                a = str[36:length(str)]
                a = replace(a," "=>"")
                home[5] = a
            end
            if length(str) > 45 
                a = str[36:45]
                a = replace(a," "=>"")
                home[5] = a
            end
            #FIELD6
            if length(str) > 45 
                a = str[46:length(str)]
                a = replace(a," "=>"")
                home[6] = a
            end
            tempHome = homeDir
            z = length(homeDir[:,1])+1
            homeDir = Array{Union{Missing, String}}(missing, z, 6)
            homeDir[1:z-1,:] = tempHome
            homeDir[z,:] = home
            for i = 1:length(homeDir[:,1])
                for j = 1:length(homeDir[1,:])
                    if string(homeDir[i,j]) == "missing"
                        homeDir[i,j] = ""
                    end
                end
            end
        end
    end
    return homeDir
end

#ALL POSSIBLE PARAMETERS THAT CAN BE DEFINED
function params(problem::String,hDirLine)
    orig = ""
    replaceR = ""
    str = hDirLine[1]
    if str == "IE"
        orig = hDirLine[2]
        replaceR = hDirLine[4]
    end
    if str == "IR"
        orig = hDirLine[2]
        replaceR = hDirLine[3]
    end
    if str == "IA"
        orig = hDirLine[2]
        replaceR = hDirLine[3]*"+"*hDirLine[4]
    end
    if str == "IS"
        orig = hDirLine[2]
        replaceR = hDirLine[4]*"-"*hDirLine[3]
    end
    if str == "IM"
        orig = hDirLine[2]
        replaceR = hDirLine[3]*"*"*hDirLine[4]
    end
    if str == "ID"
        orig = hDirLine[2]
        replaceR = hDirLine[4]*"/"*hDirLine[3]
    end
    if str == "I="
        orig = hDirLine[2]
        replaceR = hDirLine[3]
    end
    if str == "I+"
        orig = hDirLine[2]
        replaceR = hDirLine[3]*"+"*hDirLine[5]
    end
    if str == "I-"
        orig = hDirLine[2]
        replaceR = hDirLine[3]*"-"*hDirLine[5]
    end
    if str == "I*"
        orig = hDirLine[2]
        replaceR = hDirLine[3]*"*"*hDirLine[5]
    end
    if str == "I/"
        orig = hDirLine[2]
        replaceR = hDirLine[3]*"/"*hDirLine[5]
    end
    if str == "RE"
        orig = hDirLine[2]
        replaceR = hDirLine[4]
    end
    if str == "RI"
        orig = hDirLine[2]
        replaceR = hDirLine[3]
    end
    if str == "RA"
        orig = hDirLine[2]
        replaceR = hDirLine[3]*"+"*hDirLine[4]
    end
    if str == "RS"
        orig = hDirLine[2]
        replaceR = hDirLine[4]*"-"*hDirLine[3]
    end
    if str == "RM"
        orig = hDirLine[2]
        replaceR = hDirLine[3]*"*"*hDirLine[4]
    end
    if str == "RD"
        orig = hDirLine[2]
        replaceR = hDirLine[4]*"/"*hDirLine[3]
    end
    if str == "RF"
        orig = hDirLine[2]
        replaceR = hDirLine[3]*"("*hDirLine[4]*")"
    end
    if str == "R="
        orig = hDirLine[2]
        replaceR = hDirLine[3]
    end
    if str == "R+"
        orig = hDirLine[2]
        replaceR = hDirLine[3]*"+"*hDirLine[5]
    end
    if str == "R-"
        orig = hDirLine[2]
        replaceR = hDirLine[3]*"-"*hDirLine[5]
    end
    if str == "R*"
        orig = hDirLine[2]
        replaceR = hDirLine[3]*"*"*hDirLine[5]
    end
    if str == "R/"
        orig = hDirLine[2]
        replaceR = hDirLine[3]*"/"*hDirLine[5]
    end
    if str == "R("
        orig = hDirLine[2]
        replaceR = hDirLine[3]*"("*hDirLine[5]*")"
    end
    #FIX FOR ARRAY??
    if str == "AE"
        orig = hDirLine[2]
        replaceR = hDirLine[4]
        println("Should this be modified for an array?")
    end
    #FIX FOR ARRAY??
    if str == "AI"
        orig = hDirLine[2]
        replaceR = hDirLine[3]
        println("Should this be modified for an array?")
    end
    #FIX FOR ARRAY??
    if str == "AA"
        orig = hDirLine[2]
        replaceR = hDirLine[3]*"+"*hDirLine[4]
        println("Should this be modified for an array?")
    end
    #FIX FOR ARRAY??
    if str == "AS"
        orig = hDirLine[2]
        replaceR = hDirLine[4]*"-"*hDirLine[3]
        println("Should this be modified for an array?")
    end
    #FIX FOR ARRAY??
    if str == "AM"
        orig = hDirLine[2]
        replaceR = hDirLine[3]*"*"*hDirLine[4]
        println("Should this be modified for an array?")
    end
    #FIX FOR ARRAY??
    if str == "AD"
        orig = hDirLine[2]
        replaceR = hDirLine[4]*"/"*hDirLine[3]
        println("Should this be modified for an array?")
    end
    #FIX FOR ARRAY??
    if str == "AF"
        orig = hDirLine[2]
        replaceR = hDirLine[3]*"("*hDirLine[4]*")"
        println("Should this be modified for an array?")
    end
    #FIX FOR ARRAY??
    if str == "A="
        orig = hDirLine[2]
        replaceR = hDirLine[3]
        println("Should this be modified for an array?")
    end
    #FIX FOR ARRAY??
    if str == "A+"
        orig = hDirLine[2]
        replaceR = hDirLine[3]*"+"*hDirLine[5]
        println("Should this be modified for an array?")
    end
    #FIX FOR ARRAY??
    if str == "A-"
        orig = hDirLine[2]
        replaceR = hDirLine[3]*"-"*hDirLine[5]
        println("Should this be modified for an array?")
    end
    #FIX FOR ARRAY??
    if str == "A*"
        orig = hDirLine[2]
        replaceR = hDirLine[3]*"*"*hDirLine[5]
        println("Should this be modified for an array?")
    end
    #FIX FOR ARRAY??
    if str == "A/"
        orig = hDirLine[2]
        replaceR = hDirLine[3]*"/"*hDirLine[5]
        println("Should this be modified for an array?")
    end
    #FIX FOR ARRAY??
    if str == "A("
        orig = hDirLine[2]
        replaceR = hDirLine[3]*"("*hDirLine[5]*")"
        println("Should this be modified for an array?")
    end
    return orig, replaceR
end

#REPLACE ALL PRE-DEFINED PARAMETERS
function initParams(problem::String,lineVar,line,homeDir)
    for i = 1:lineVar
        reps = params(problem,homeDir[i,:])
        orig = reps[1]
        replaceR = reps[2]
        if reps[1] != ""
            for j = 1:line
                for k = 2:6
                    if homeDir[j,k] == orig
                        homeDir[j,k] = replace(homeDir[j,k],orig=>replaceR)
                    else
                        homeDir[j,k] = replace(homeDir[j,k],"("*orig=>"("*replaceR)
                    end
                end
            end
        end
    end
    return homeDir
end

#DO LOOP PARSER -- NEEDS HELP, DOESNT WORK
function doLoop(problem::String,startDir,endDir,homeDir)
    doLoopMat=Array{Union{Missing, String}}(missing, endDir-startDir+1, 6)
    beg=""
    dun=""
    indice=""
    step=""
    for i = startDir:endDir
        beg = startDir[3]
        dun = startDir[5]
        indice = startDir[2]
        if homeDir[i,1] == "DI"
            step = homeDir[i,2]
        end
        reps = params(problem,homeDir[i,:])
        orig = reps[1]
        replaceR = reps[2]
        if reps[1] != ""
            for j = 1:line
                for k = 2:6
                    if homeDir[j,k] == orig
                        b = split(homeDir[j,k],"(")
                        if length(b) > 1
                            b[2] = replace(b[2],orig=>replaceR)
                            homeDir[j,k] = b[1]*b[2]
                        else
                            homeDir[j,k] = replaceR
                        end
                    end
                end
            end
        end
        # doLoopMat[startDir+i-1,:] = homeDir[i,:]
    end
    #IF WE HAVE NESTED DO LOOPS, "OD I" [I=indice], marks end of a nested do-loop indexed by i
    #FIX IF THIS BECOMES AN ISSUE
    return beg, dun, step, indice#, doLoopMat
end

#CREATE GROUPS
function groupMake(problem::String,lineGroup,lineBoundz,lineConstz,elemTot,homeDir)
    gro = Array{Union{Missing, String}}(missing, 5+elemTot, 1)
    gropz = ["";"";"";""]
    doLoop = 0
    beg = 0
    dun = 0
    indice = ""
    step = 0
    startDir = 0
    endDir = 0
    if lineConstz == 0
        lineConstz = lineBoundz
    end
    for i = lineGroup:lineConstz
        if homeDir[i,1] == "DO"
            startDir = i
            beg = eval(Meta.parse(homeDir[i,3]))
            dun = eval(Meta.parse(homeDir[i,5]))
            indice = homeDir[i,2]
            doLoop = 1
        end
        if homeDir[i,1] == "DI"
            step = eval(Meta.parse(homeDir[i,2]))
        end
        if homeDir[i,1] == "ND"
            endDir = i
        end
    end
        for i = lineGroup:lineConstz
            if homeDir[i,1] == "N" || homeDir[i,1] == "G" || homeDir[i,1] == "L" || homeDir[i,1] == "E"
                a=""
                if homeDir[i,3] != "'SCALE'"
                gropz = Array{Union{Missing, String}}(missing, 5+elemTot, 1) 
                gropz[1] = homeDir[i,2]
                if homeDir[i,3] != ""
                    b = homeDir[i,3] 
                end
                if homeDir[i,4] != ""
                    a = homeDir[i,4]*b
                end
                fillUp=0
                if a!=""
                    for d = 6:length(gropz)
                        if string(gropz[d]) == "missing" && fillUp == 0
                            gropz[d] = a
                            fillUp=1
                        end
                    end
                end
                end
                gro = [gro gropz]
            end
            if homeDir[i,1] == "XN" || homeDir[i,1] == "XL" || homeDir[i,1] == "XG" || homeDir[i,1] == "XE"
                if doLoop == 0    
                    a=""
                    gropz = Array{Union{Missing, String}}(missing, 5+elemTot, 1)
                    gropz[1] = homeDir[i,2]
                    if homeDir[i,3] != "" && homeDir[i,3] != "'SCALE'"
                        a = homeDir[i,3]
                    end
                    if homeDir[i,4] != "" && homeDir[i,3] != "'SCALE'"
                        a = homeDir[i,4]*a
                    end
                    fillUp=0
                    if a != ""
                        for d = 6:length(gropz)
                            if string(gropz[d]) == "missing" && fillUp == 0
                                gropz[d] = a
                                fillUp=1
                            end
                        end
                    end
                    gro = [gro gropz]
                end
            end
            #NO IDEA IF THIS IS CORRECT -- FIND EXAMPLE
            if homeDir[i,1] == "ZN" || homeDir[i,1] == "ZL" || homeDir[i,1] == "ZG" || homeDir[i,1] == "ZE"
                if doLoop == 0
                    a=""
                    gropz = Array{Union{Missing, String}}(missing, 5+elemTot, 1)
                    gropz[1] = homeDir[i,2]
                    if homeDir[i,3] != "" && homeDir[i,3] != "'SCALE'"
                        a = homeDir[i,3]
                    end
                    if homeDir[i,5] != "" && homeDir[i,3] != "'SCALE'"
                        a = homeDir[i,5]*a
                    end
                    fillUp=0
                    if a != ""
                        for d = 6:length(gropz)
                            if string(gropz[d]) == "missing" && fillUp == 0
                                gropz[d] = a
                                fillUp=1
                            end
                        end
                    end
                    gro = [gro gropz]
                end
            end
            #NOT SURE IF THIS IS CODED RIGHT -- FIND AN EXAMPLE
            if homeDir[i,1] == "DN" || homeDir[i,1] == "DG" || homeDir[i,1] == "DL" || homeDir[i,1] == "DE"
                if homeDir[i,3] != "'SCALE'"
                    gropz = Array{Union{Missing, String}}(missing, 5+elemTot, 1) 
                    gropz[1] = homeDir[i,2]
                    gropz[2] = homeDir[i,4]*homeDir[i,3]+homeDir[i,6]*homeDir[i,5]
                    gro = [gro gropz]
                end
            end
        end
        for i = lineGroup:lineConstz
            if i<startDir && i>endDir
            if homeDir[i,3] == "'SCALE'"
                for j = 1:length(gro[1,:])
                    z = replace(homeDir[i,2],indice=>string(l))
                    if gro[1,j] == z
                        gro[5,j] = string(1/Meta.parse(homeDir[i,4]))
                    end
                end
            end
            end
        end
        if doLoop == 1
            # startDir = convert(Float64,startDir)
            # endDir = convert(Float64,endDir)
            # doLoopOut = doLoop(problem,startDir,endDir,homeDir)
            # beg = doLoopOut[1]
            # dun = doLoopOut[2]
            # step = doLoopOut[3]
            # indice = doLoopOut[4]
            doMat = homeDir[startDir:endDir,1:6]
            if step == 0
                steps = 1
            else
                steps = step
            end
            for i = 1:steps:(endDir-startDir+1)
                outParams = params(problem,startDir+i-1)
                orig = outParams[1]
                replaceR = outParams[2] 
                if orig != ""
                    for j = 1:line
                        for k = 2:6
                            doMat[j,k] = replace(doMat[j,k],"("*orig*")"=>"("*replaceR*")")
                        end
                    end
                end
            end
            for i = 1:steps:(endDir-startDir+1)
                a=""
                if doMat[i,1] == "XN" || doMat[i,1] == "XL" || doMat[i,1] == "XG" || doMat[i,1] == "XE"
                    if doMat[i,3] != "'SCALE'"
                    for l = beg:dun
                        gropz = Array{Union{Missing, String}}(missing, 5+elemTot, 1)
                        gropz[1] = replace(doMat[i,2],indice=>string(l))
                        if doMat[i,3] != "" && doMat[i,3] != "'SCALE'"
                            a = replace(doMat[i,3],indice=>string(l))
                            b=split(a,"(")
                            c=eval(Meta.parse("("*b[2]))
                            a=b[1]*"("*string(c)*")"
                        end
                        if doMat[i,4] != "" && doMat[i,3] != "'SCALE'"
                            a = doMat[i,4]*a
                        end
                        fillUp=0
                        if a != ""
                            for d = 6:length(gropz)
                                if string(gropz[d]) == "missing" && fillUp == 0
                                    gropz[d] = a
                                    fillUp=1
                                end
                            end
                        end
                        gro = [gro gropz]
                    end
                    end
                end
                #NO IDEA IF THIS IS CORRECT -- FIND EXAMPLE
                if doMat[i,1] == "ZN" || doMat[i,1] == "ZL" || doMat[i,1] == "ZG" || doMat[i,1] == "ZE"
                    if doMat[i,3] != "'SCALE'"
                    for l = beg:dun
                        gropz = Array{Union{Missing, String}}(missing, 5+elemTot, 1)
                        gropz[1] = replace(doMat[i,2],indice=>string(l))
                        if doMat[i,3] != "" && doMat[i,3] != "'SCALE'"
                            a = replace(doMat[i,3],indice=>string(l))
                            b=split(a,"(")
                            c=eval(Meta.parse("("*b[2]))
                            a=b[1]*"("*string(c)*")"
                        end
                        if doMat[i,5] != "" && doMat[i,3] != "'SCALE'"
                            a = doMat[i,5]*a
                        end
                        fillUp=0
                        if a != ""
                            for d = 6:length(gropz)
                                if string(gropz[d]) == "missing" && fillUp == 0
                                    gropz[d] = a
                                    fillUp=1
                                end
                            end
                        end
                        gro = [gro gropz]
                    end
                    end
                end
            end
            for i = 1:steps:(endDir-startDir+1)
                if doMat[i,3] == "'SCALE'"
                    for j = 1:length(gro[1,:])
                        for l=beg:dun
                        z = replace(doMat[i,2],indice=>string(l))
                        if string(gro[1,j]) == z
                            gro[5,j] = string(1/Meta.parse(doMat[i,4]))
                        end
                        end
                    end
                end
            end
        end
        for i = 1:length(gro[1,:])
            for j = 1:length(gro[:,1])
                if string(gro[j,i]) == "missing"
                    gro[j,i] = ""
                end
            end
        end
    return gro
end

#ASSIGN ELEMENTS TO GROUPS
function groupUse(problem::String,lineGroUse,lineObjBound,gro,homeDir)
    doLoop = 0
    doMat = ["","","","","",""]
    beg = 0
    dun = 0
    indice = ""
    startDir=0
    endDir=0
    step = 0
    for t = lineGroUse:lineObjBound
        if homeDir[t,2] == "'DEFAULT'"
            defau = homeDir[t,3]
            for i = 1:length(gro[1,:])
                gro[3,i] = defau
            end
        end
    end
    for t=lineGroUse:lineObjBound
        if homeDir[t,1] == "DO"
            startDir = t
            beg = eval(Meta.parse(homeDir[t,3]))
            dun = eval(Meta.parse(homeDir[t,5]))
            indice = homeDir[t,2]
            doLoop = 1
        end
        if homeDir[t,1] == "DI"
            step = eval(Meta.parse(homeDir[t,2]))
        end
        if homeDir[t,1] == "ND"
            endDir = t
        end
    end
    for t = lineGroUse:lineObjBound
            if homeDir[t,1] == "T"
                for k = 1:length(gro[1,:])
                    if string(gro[1,k]) == homeDir[t,2]
                        gro[3,k] = homeDir[t,3]
                    end
                end
            end
            if homeDir[t,1] == "E"
                for k = 1:length(gro[1,:])
                    fillUp = 0
                    if string(gro[1,k]) == homeDir[t,2]
                        for i = 6:length(gro[:,1])
                            if gro[i,k] == "" && fillUp == 0
                                gro[i,k] = homeDir[t,3]
                                if homeDir[t,4] == ""
                                    gro[i,k] = "1"*gro[i,k]
                                else
                                    gro[i,k] = homeDir[t,4]*gro[i,k]
                                end
                                if homeDir[t,5] != ""
                                    gro[i+1,k] = homeDir[t,5]
                                end
                                if homeDir[t,6] == ""
                                    gro[i,k+1] = "1"*gro[i,k+1]
                                else
                                    gro[i,k+1] = homeDir[t,6]*gro[i,k+1]
                                end
                                fillUp = 1
                            end
                        end
                    end
                end
            end
            if homeDir[t,1] == "P"
                for k = 1:length(gro[1,:])
                    fillUp = 0
                    if string(gro[1,k]) == homeDir[t,2]
                        for i = 6:length(gro[:,1])
                            if gro[i,k] == "" && fillUp == 0
                                gro[i,k] = homeDir[t,4]
                                if homeDir[t,5] != ""
                                    gro[i+1,k] = homeDir[t,6]
                                end
                                fillUp = 1
                            end
                        end
                    end
                end
            end
            if t<startDir || t>endDir
                if homeDir[t,1] == "ZE"
                    for k = 1:length(gro[1,:])
                        if string(gro[1,k]) == homeDir[t,2]
                            fillUp = 0
                            for j = 6:length(gro[:,1])
                                if gro[j,k] == "" && fillUp == 0
                                    gro[j,k] = homeDir[t,3]
                                    if homeDir[t,5] != ""
                                        gro[j,k] = homeDir[t,5]*gro[j,k]
                                    end
                                    fillUp = 1
                                end
                            end
                        end
                    end
                end
                #CHECK FOR ZP EXAMPLE -- NOT SURE IF CODED RIGHT
                if homeDir[t,1] == "ZP"
                    for k = 1:length(gro[1,:])
                        if string(gro[1,k]) == homeDir[t,2]
                            fillUp = 0
                            for j = 6:length(gro[:,1])
                                if gro[j,k] == "" && fillUp == 0
                                    if homeDir[t,5] == ""
                                        gro[j,k]=1
                                    else
                                        gro[j,k] = homeDir[t,5]
                                    end
                                    fillUp = 1
                                end
                            end
                        end
                    end
                end
                if homeDir[t,1] == "XE"
                    for k = 1:length(gro[1,:])
                        if string(gro[1,k]) == homeDir[t,2]
                            fillUp = 0
                            for j = 6:length(gro[:,1])
                                if gro[j,k] == "" && fillUp == 0
                                    gro[j,k] = homeDir[t,3]
                                    if homeDir[t,4] != ""
                                        gro[j,k] = homeDir[t,4]*"*"*gro[j,k]
                                    end
                                    if homeDir[t,5] != ""
                                        gro[j+1,k] = homeDir[t,5]
                                        if homeDir[t,6] != ""
                                            gro[j+1,k] = homeDir[t,6]*"*"*gro[j+1,k]
                                        end
                                    end
                                    fillUp = 1
                                end
                            end
                        end
                    end
                end
                #CHECK FOR XP EXAMPLE -- NOT SURE IF CODED RIGHT
                if homeDir[t,1] == "XP"
                    for k = 1:length(gro[1,:])
                        if string(gro[1,k]) == homeDir[t,2]
                            fillUp = 0
                            for j = 6:length(gro[:,1])
                                if gro[j,k] == "" && fillUp == 0
                                    if homeDir[t,4] != ""
                                        gro[j,k] = homeDir[t,4]*gro[j,k]
                                    else
                                        gro[j,k] = "1"
                                    end
                                    if homeDir[t,6] != ""
                                        gro[j+1,k] = homeDir[t,6]*gro[j+1,k]
                                    else
                                        gro[j+1,k] = "1"
                                    end
                                    fillUp = 1
                                end
                            end
                        end
                    end
                end
            end
    end
    doMat = homeDir[startDir:endDir,1:6]
    if doLoop == 1
            if step == 0
                steps = 1
            else
                steps = step
            end
            for i = 1:steps:(endDir-startDir+1)
                outParams = params(problem,doMat[i,:])
                orig = outParams[1]
                replaceR = outParams[2]
                if orig != ""
                    for j = 1:length(doMat[:,1])
                        for k = 2:6
                            doMat[j,k] = replace(doMat[j,k],"("*orig*")"=>"("*replaceR*")")
                            if doMat[j,k] == orig
                                doMat[j,k] = replaceR
                            end
                        end
                    end
                end
            end
        for t = 1:(endDir-startDir+1)
            if doMat[t,1] == "ZE"
                for l = beg:dun
                    a = replace(doMat[t,2],indice=>string(l))
                    b=split(a,"(")
                    if length(b)>1
                        c=eval(Meta.parse("("*b[2]))
                        d=b[1]*"("*string(c)*")"
                    else
                        d=a
                    end
                for k = 1:length(gro[1,:])
                    if string(gro[1,k]) == d
                        fillUp = 0
                        for j = 6:length(gro[:,1])
                            if gro[j,k] == "" && fillUp == 0
                                a = replace(doMat[t,3],indice=>string(l))
                                b=split(a,"(")
                                c=eval(Meta.parse("("*b[2]))
                                a=b[1]*"("*string(c)*")"
                                gro[j,k] = a
                                if doMat[t,5] != ""
                                    a = replace(doMat[t,5],indice=>string(l))
                                    b=split(a,"(")
                                    if length(b)>1
                                        c=eval(Meta.parse("("*b[2]))
                                        a=b[1]*"("*string(c)*")"
                                    end
                                    gro[j,k] = a*"*"*gro[j,k]
                                end
                                fillUp = 1
                            end
                        end
                    end
                end
                end
            end
            #CHECK FOR ZP EXAMPLE -- NOT SURE IF CODED RIGHT
            if doMat[t,1] == "ZP"
                for l = beg:dun
                    a = replace(doMat[t,2],indice=>string(l))
                    b=split(a,"(")
                    c=eval(Meta.parse("("*b[2]))
                    d=b[1]*"("*string(c)*")"
                for k = 1:length(gro[1,:])
                    if string(gro[1,k]) == d
                        fillUp = 0
                        for j = 6:length(gro[:,1])
                            if gro[j,k] == "" && fillUp == 0
                                # a = replace(doMat[t,3],indice=>string(l))
                                # b=split(a,"(")
                                # c=eval(Meta.parse("("*b[2]))
                                # a=b[1]*"("*string(c)*")"
                                gro[j,k] = doMat[t,5]
                                fillUp = 1
                            end
                        end
                    end
                end
                end
            end
            if doMat[t,1] == "XE"
                for l = beg:dun
                    a = replace(doMat[t,2],indice=>string(l))
                    b=split(a,"(")
                    c=eval(Meta.parse("("*b[2]))
                    d=b[1]*"("*string(c)*")"
                for k = 1:length(gro[1,:])
                    if string(gro[1,k]) == d
                        fillUp = 0
                        for j = 6:length(gro[:,1])
                            if gro[j,k] == "" && fillUp == 0
                                a = replace(doMat[t,3],indice=>string(l))
                                b=split(a,"(")
                                c=eval(Meta.parse("("*b[2]))
                                a=b[1]*"("*string(c)*")"
                                gro[j,k] = a
                                if doMat[t,4] != ""
                                    gro[j,k] = doMat[t,4]*"*"*gro[j,k]
                                end
                                if doMat[t,5] != ""
                                    a = replace(doMat[t,5],indice=>string(l))
                                    b=split(a,"(")
                                    c=eval(Meta.parse("("*b[2]))
                                    a=b[1]*"("*string(c)*")"
                                    gro[j+1,k] = a
                                    if doMat[t,6] != ""
                                        gro[j+1,k] = doMat[t,6]*"*"*gro[j+1,k]
                                    end
                                end
                                fillUp = 1
                            end
                        end
                    end
                end
                end
            end
            #CHECK FOR XP EXAMPLE -- NOT SURE IF CODED RIGHT
            if doMat[t,1] == "XP"
                for l = beg:dun
                    a = replace(doMat[t,2],indice=>string(l))
                    b=split(a,"(")
                    c=eval(Meta.parse("("*b[2]))
                    d=b[1]*"("*string(c)*")"
                for k = 1:length(gro[1,:])
                    if string(gro[1,k]) == d
                        fillUp = 0
                        for j = 6:length(gro[:,1])
                            if gro[j,k] == "" && fillUp == 0
                                # a = replace(doMat[t,3],indice=>string(l))
                                # b=split(a,"(")
                                # c=eval(Meta.parse("("*b[2]))
                                # a=b[1]*"("*string(c)*")"
                                # gro[j,k] = a
                                if doMat[t,4] != ""
                                    gro[j,k] = doMat[t,4]*gro[j,k]
                                else
                                    gro[j,k] = "1"
                                end
                                if doMat[t,6] != ""
                                    gro[j+1,k] = doMat[t,6]*gro[j+1,k]
                                else
                                    gro[j+1,k] = "1"
                                end
                                fillUp = 1
                            end
                        end
                    end
                end
                end
            end
        end
    end
    return gro
end

#SET GROUP TYPE
function groupType(problem::String,lineGroType,lineGroUse,gro,homeDir)
    beg = 0
    dun = 0
    indice = ""
    line = 0
    for i = lineGroType:lineGroUse
            if homeDir[i,1] == "GV"
                for k = 1:length(gro[1,:])
                    if gro[3,k] == homeDir[i,2]
                        gro[4,k] = homeDir[i,3]
                    end
                end
            end
        #DO WE NEED TO FIND AND STORE GROUP PARAMETERS?? 
        #WHAT ARE THEY USED FOR??
    end
    return gro
end

#SET CONSTANTS FOR FUNCTION
function constAssig(problem::String,lineConstz::Int64,lineBoundz::Int64,gro,homeDir)
        doLoop = 0
        step = 1
        startDir=0
        endDir=0
        beg=0
        doMat = ["","","","","",""]
        dun=0
        indice = ""
        if lineConstz != 0
        #DEFAULT CONSTANT SET -- WHAT??
        for t = lineConstz:lineBoundz
            #NOT SURE WHAT TO DO WITH DEFAULT CONSTANTS
            if homeDir[t,3] == "'DEFAULT'"
                defau = homeDir[t,4]
                println("HELP WHAT DO I DO WITH DEFAULT CONSTANTS")
            end
        end
        for t=lineConstz:lineBoundz
            if homeDir[t,1] == "DO"
                startDir = t
                beg = eval(Meta.parse(homeDir[t,3]))
                dun = eval(Meta.parse(homeDir[t,5]))
                indice = homeDir[t,2]
                doLoop = 1
            end
            if homeDir[t,1] == "DI"
                step = eval(Meta.parse(homeDir[t,2]))
            end
            if homeDir[t,1] == "ND"
                endDir = t
            end
        end
        for i =lineConstz:lineBoundz
            if homeDir[i,1] == "" && homeDir[i,3] != "'DEFAULT'"
                fillUp = 0
                fillUp2 = 0
                for k = 2:length(gro[1,:])
                    if string(gro[1,k]) == string(homeDir[i,3])
                        for j = 6:length(gro[:,1])
                            if gro[j,k] == "" && fillUp == 0
                                gro[j,k] = string(-1*Meta.parse(homeDir[i,4]))
                                fillUp = 1
                            end
                        end
                    end
                    if string(gro[1,k]) == string(homeDir[i,5])
                        for j = 6:length(gro[:,1])
                            if gro[j,k] == "" && fillUp == 0
                                gro[j,k] = string(-1*Meta.parse(homeDir[i,6]))
                                fillUp2 = 1
                            end
                        end
                    end
                end
            end
            if homeDir[i,1] == "X" && homeDir[i,3] != "'DEFAULT'"
                fillUp = 0
                fillUp2 = 0
                for k = 2:length(gro[1,:])
                    if string(gro[1,k]) == string(homeDir[i,3])
                        for j = 6:length(gro[:,1])
                            if gro[j,k] == "" && fillUp == 0
                                gro[j,k] = string(-1*Meta.parse(homeDir[i,4]))
                                fillUp = 1
                            end
                        end
                    end
                    if string(gro[1,k]) == string(homeDir[i,5])
                        for j = 6:length(gro[:,1])
                            if gro[j,k] == "" && fillUp == 0
                                gro[j,k] = string(-1*Meta.parse(homeDir[i,6]))
                                fillUp2 = 1
                            end
                        end
                    end
                end
            end
            if homeDir[i,1] == "Z" && homeDir[i,3] != "'DEFAULT'"
                fillUp = 0
                for k = 2:length(gro[1,:])
                    if string(gro[1,k]) == string(homeDir[i,3])
                        for j = 6:length(gro[:,1])
                            if gro[j,k] == "" && fillUp == 0
                                gro[j,k] = string(-1*Meta.parse(homeDir[i,5]))
                                fillUp = 1
                            end
                        end
                    end
                end
            end
        end
        if doLoop == 1
        doMat = homeDir[startDir:endDir,1:6]
        for i = 1:length(doMat[:,1])
            if doMat[i,1] == "X" && doMat[i,3] != "'DEFAULT'"
                for d = beg:dun
                    z=replace(doMat[i,3],indice=>string(d))
                    a=split(z,"(")
                    if length(a) > 1
                        q = eval(Meta.parse("("*a[2]))
                        w = a[1]*"("*string(q)*")"
                    else
                        w = z
                    end
                    z=replace(doMat[i,5],indice=>string(d))
                    a=split(z,"(")
                    if length(a) > 1
                        q = eval(Meta.parse("("*a[2]))
                        v = a[1]*"("*string(q)*")"
                    else 
                        v = z
                    end
                    fillUp = 0
                    fillUp2 = 0
                    for k = 2:length(gro[1,:])
                    if string(gro[1,k]) == w
                        for j = 6:length(gro[:,1])
                            if gro[j,k] == "" && fillUp == 0
                                gro[j,k] = string(-1*Meta.parse(doMat[i,4]))
                                fillUp = 1
                            end
                        end
                    end
                    if string(gro[1,k]) == v
                        for j = 6:length(gro[:,1])
                            if gro[j,k] == "" && fillUp == 0
                                gro[j,k] = string(-1*Meta.parse(doMat[i,6]))
                                fillUp2 = 1
                            end
                        end
                    end
                    end
                end
            end
            if homeDir[i,1] == "Z" && homeDir[i,3] != "'DEFAULT'"
                fillUp = 0
                for k = 2:length(gro[1,:])
                    for d = beg:dun
                    z=replace(doMat[i,3],indice=>string(d))
                    a=split(z,"(")
                    q = eval(Meta.parse("("*a[2]))
                    w = z[1]*"("*string(q)*")"
                    if string(gro[1,k]) == w
                        for j = 6:length(gro[:,1])
                            if gro[j,k] == "" && fillUp == 0
                                gro[j,k] = string(-1*Meta.parse(doMat[i,5]))
                                fillUp = 1
                            end
                        end
                    end
                    end
                end
            end
        end
        end
        end
    return gro
end

#ASSIGN VARIABLES TO ELEMENTS
function eleUseSet(problem::String,lineGroType::Int64,lineEleUse::Int64,varTot,homeDir)
    elementz = ["";"";"";""]
    elems = Array{Union{Missing, String}}(missing, varTot+3, 1)
    doLoop = 0
    beg = 0
    dun = 0
    indice = ""
    step=0
    startDir=0
    endDir=0
    doMat = ["","","","","",""]
    for t=lineEleUse:lineGroType
        if homeDir[t,1] == "DO"
            startDir = t
            beg = eval(Meta.parse(homeDir[t,3]))
            dun = eval(Meta.parse(homeDir[t,5]))
            indice = homeDir[t,2]
            doLoop = 1
        end
        if homeDir[t,1] == "DI"
            step = eval(Meta.parse(homeDir[t,2]))
        end
        if homeDir[t,1] == "ND"
            endDir = t
        end
    end
    if doLoop ==1
        doMat = homeDir[startDir:endDir,1:6]
    end
    for i = lineEleUse:lineGroType
            if homeDir[i,1] == "T" 
                extra=Array{Union{Missing, String}}(missing, varTot+3, 1)
                extra[1] = homeDir[i,2]
                extra[2] = homeDir[i,3]
                elems = [elems extra]
            end
            if homeDir[i,1] == "V" 
                for j = 1:length(elems[1,:])
                    if elems[1,j] == homeDir[i,2]
                        fillUp = 0
                        for k = 4:length(elems[:,1])
                            if string(elems[k,j]) == "missing" && fillUp == 0
                                elems[k,j] = homeDir[i,3]
                                elems[k+1,j] = homeDir[i,5]
                                fillUp = 1
                            end
                        end
                    end
                end
            end
            if homeDir[i,1] == "P" 
                for j = 1:length(elems[1,:])
                    if elems[1,j] == homeDir[i,2]
                        for k = 4:length(elems[:,1])
                            if string(elems[k,j]) == "missing"
                                elems[k,j] = homeDir[i,3]
                                elems[k+1,j] = homeDir[i,4]
                                if homeDir[i,5] != ""
                                    elems[k+2,j] = homeDir[i,5]
                                    elems[k+3,j] = homeDir[i,6]
                                end
                            end
                        end
                    end
                end
            end
        if i<startDir || i>endDir
            if homeDir[i,1] == "XT" && homeDir[i,2] != "'DEFAULT'"
                    elementz=Array{Union{Missing, String}}(missing, varTot+3, 1)
                    elementz[1] = homeDir[i,2]
                    elementz[2] = homeDir[i,3]
                    elems = [elems elementz]
            end
            if homeDir[i,1] == "ZV" 
                    z = homeDir[i,2]
                    success=0
                    for j = 1:length(elems[1,:])
                        if string(elems[1,j]) == string(z)
                            success=1
                            fillUp=0
                            for k = 4:length(elems[:,1])
                                if string(elems[k,j]) == "missing" && fillUp ==0
                                    fillUp = 1
                                    elems[k,j] = homeDir[i,3]
                                    q = split(homeDir[i,5],"(")
                                    if length(q)>1
                                        p = eval(Meta.parse("("*q[2]))
                                        w = q[1]*"("*string(p)*")"
                                    else
                                        w = homeDir[i,5]
                                    end
                                    elems[k+1,j] = w
                                end
                            end
                        end
                    end
                    if success==0
                        elementz=Array{Union{Missing, String}}(missing, varTot+3, 1)
                        elementz[1]=z
                        elementz[4] = homeDir[i,3]
                        elementz[5] = homeDir[i,5]
                        elems = [elems elementz]
                    end
            end
            if homeDir[i,1] == "ZP" 
                    z = homeDir[i,2]
                    success=0
                    for j = 1:length(elems[1,:])
                        fillUp=0
                        if string(elems[1,j]) == z
                            success=1
                            for k = 4:length(elems[:,1])
                                if string(elems[k,j]) == "missing" && fillUp ==0
                                    elems[k,j] = homeDir[i,3]
                                    elems[k+1,j] = homeDir[i,5]
                                    fillUp=1
                                end
                            end
                        end
                    end
                    if success==0
                        elementz=Array{Union{Missing, String}}(missing, varTot+3, 1)
                        elementz[1]=z
                        elementz[4] = homeDir[i,3]
                        elementz[5,j] = homeDir[i,5]
                        elems = [elems elementz]
                    end
            end
            if homeDir[i,1] == "XP" 
                    z = homeDir[i,2]
                    success=0
                    for j = 1:length(elems[1,:])
                        if elems[1,j] == z
                            fillUp=0
                            success=1
                            for k = 4:length(elems[:,1])
                                if string(elems[k,j]) == "missing" && fillUp==0
                                    fillUp=1
                                    elems[k,j] = homeDir[i,3]
                                    elems[k+1,j] = homeDir[i,4]
                                    if homeDir[i,5] != ""
                                        elems[k+2,j] = homeDir[i,5]
                                        elems[k+3,j] = homeDir[i,6]
                                    end
                                end
                            end
                        end
                    end
                    if success==0
                        elementz=Array{Union{Missing, String}}(missing, varTot+3, 1)
                        elementz[1]=z
                        elementz[4] = homeDir[i,3]
                        elementz[5] = homeDir[i,4]
                        if homeDir[i,5] != ""
                            elementz[6] = homeDir[i,5]
                            elementz[7] = homeDir[i,6]
                            elems = [elems elementz]
                        end
                    end
            end
        end
    end
    if doLoop == 1
        if step == 0
            steps = 1
        else
            steps = step
        end
        for i = 1:steps:(endDir-startDir+1)
            outParams = params(problem,doMat[i,:])
            orig = outParams[1]
            replaceR = outParams[2] 
            if orig != ""
                for j = 1:length(doMat[:,1])
                    for k = 2:6
                        doMat[j,k] = replace(doMat[j,k],"("*orig*")"=>"("*replaceR*")")
                        if doMat[j,k] == orig
                            doMat[j,k] = replaceR
                        end
                    end
                end
            end
        end
        for i = 1:(endDir-startDir+1)
            if doMat[i,1] == "XT" && doMat[i,2] != "'DEFAULT'"
                for d = beg:dun
                    elementz=Array{Union{Missing, String}}(missing, varTot+3, 1)
                    a = replace(doMat[i,2],indice=>string(d))
                    b=split(a,"(")
                    if length(b)>1
                        c=eval(Meta.parse("("*b[2]))
                        z=b[1]*"("*string(c)*")"
                    else
                        z=a
                    end
                    elementz[1] = z
                    elementz[2] = doMat[i,3]
                    elems = [elems elementz]
                end
            end
            if doMat[i,1] == "ZV" 
                for d = beg:dun
                    a = replace(doMat[i,2],indice=>string(d))
                    b=split(a,"(")
                    if length(b)>1
                        c=eval(Meta.parse("("*b[2]))
                        z=b[1]*"("*string(c)*")"
                    else
                        z=a
                    end
                    success=0
                    for j = 1:length(elems[1,:])
                        if string(elems[1,j]) == string(z)
                            success=1
                            fillUp=0
                            for k = 4:length(elems[:,1])
                                if string(elems[k,j]) == "missing" && fillUp ==0
                                    fillUp = 1
                                    elems[k,j] = doMat[i,3]
                                    a = replace(doMat[i,5],indice=>string(d))
                                    b=split(a,"(")
                                    if length(b)>1
                                        c=eval(Meta.parse("("*b[2]))
                                        z=b[1]*"("*string(c)*")"
                                    else
                                        z=a
                                    end
                                    elems[k+1,j] = z
                                end
                            end
                        end
                    end
                    if success==0
                        elementz=Array{Union{Missing, String}}(missing, varTot+3, 1)
                        elementz[1]=z
                        elementz[4] = doMat[i,3]
                        a = replace(doMat[i,5],indice=>string(d))
                        b=split(a,"(")
                        if length(b)>1
                            c=eval(Meta.parse("("*b[2]))
                            z=b[1]*"("*string(c)*")"
                        else
                            z=a
                        end
                        elementz[5] = z
                        elems = [elems elementz]
                    end
                end
            end
            if doMat[i,1] == "ZP" 
                for d = beg:dun
                    a = replace(doMat[i,2],indice=>string(d))
                    b=split(a,"(")
                    if length(b)>1
                        c=eval(Meta.parse("("*b[2]))
                        z=b[1]*"("*string(c)*")"
                    else
                        z=a
                    end
                    success=0
                    for j = 1:length(elems[1,:])
                        fillUp=0
                        if string(elems[1,j]) == z
                            success=1
                            for k = 4:length(elems[:,1])
                                if string(elems[k,j]) == "missing" && fillUp ==0
                                    elems[k,j] = doMat[i,3]
                                    a = replace(doMat[i,5],indice=>string(d))
                                    b=split(a,"(")
                                    if length(b)>1
                                        c=eval(Meta.parse("("*b[2]))
                                        z=b[1]*"("*string(c)*")"
                                    else
                                        z=a
                                    end
                                    elems[k+1,j] = z
                                    fillUp=1
                                end
                            end
                        end
                    end
                    if success==0
                        elementz=Array{Union{Missing, String}}(missing, varTot+3, 1)
                        elementz[1]=z
                        elementz[4] = doMat[i,3]
                        a = replace(doMat[i,5],indice=>string(d))
                        b=split(a,"(")
                        if length(b)>1
                            c=eval(Meta.parse("("*b[2]))
                            z=b[1]*"("*string(c)*")"
                        else
                            z=a
                        end
                        elementz[5,j] = z
                        elems = [elems elementz]
                    end
                end
            end
            if doMat[i,1] == "XP" 
                for d = beg:dun
                    a = replace(doMat[i,2],indice=>string(d))
                    b=split(a,"(")
                    if length(b)>1
                        c=eval(Meta.parse("("*b[2]))
                        z=b[1]*"("*string(c)*")"
                    else
                        z=a
                    end
                    success=0
                    for j = 1:length(elems[1,:])
                        if elems[1,j] == z
                            fillUp=0
                            success=1
                            for k = 4:length(elems[:,1])
                                if string(elems[k,j]) == "missing" && fillUp==0
                                    fillUp=1
                                    elems[k,j] = doMat[i,3]
                                    a = replace(doMat[i,4],indice=>string(d))
                                    b=split(a,"(")
                                    if length(b)>1
                                        c=eval(Meta.parse("("*b[2]))
                                        z=b[1]*"("*string(c)*")"
                                    else
                                        z=a
                                    end
                                    elems[k+1,j] = z
                                    if doMat[i,5] != ""
                                        elems[k+2,j] = doMat[i,5]
                                        a = replace(doMat[i,6],indice=>string(d))
                                        b=split(a,"(")
                                        if length(b)>1
                                            c=eval(Meta.parse("("*b[2]))
                                            z=b[1]*"("*string(c)*")"
                                        else
                                            z=a
                                        end
                                        elems[k+3,j] = z
                                    end
                                end
                            end
                        end
                    end
                    if success==0
                        elementz=Array{Union{Missing, String}}(missing, varTot+3, 1)
                        elementz[1]=z
                        elementz[4] = doMat[i,3]
                        a = replace(doMat[i,4],indice=>string(d))
                        b=split(a,"(")
                        if length(b)>1
                            c=eval(Meta.parse("("*b[2]))
                            z=b[1]*"("*string(c)*")"
                        else
                            z=a
                        end
                        elementz[5] = z
                        if doMat[i,5] != ""
                            elementz[6] = doMat[i,5]
                            a = replace(doMat[i,6],indice=>string(d))
                            b=split(a,"(")
                        if length(b)>1
                            c=eval(Meta.parse("("*b[2]))
                            z=b[1]*"("*string(c)*")"
                        else
                            z=a
                        end
                        elementz[7] = z
                        elems = [elems elementz]
                    end
                    end
                end
            end
        end
    end
    #SET DEFAULT ELE TYPE 
    for t=lineEleUse:lineGroType
        if homeDir[t,1] == "XT" && homeDir[t,2] == "'DEFAULT'"
            for i = 1:length(elems[1,:])
                if string(elems[2,i]) == "missing"
                    elems[2,i] = homeDir[t,3]
                end
            end
        end
    end
    for i = 1:length(elems[:,1])
        for j = 1:length(elems[1,:])
            if string(elems[i,j]) == "missing"
                elems[i,j] = ""
            end
        end
    end
    elemTot = length(elems[1,:])
    return elems, elemTot
end

#INPUT MATHEMATICAL DEFINITION FOR ELEMENTS
function eleAssig(problem::String,lineEleEnd::Int64,lineGroEnd::Int64,elems,elemTot,homeDir)
        tipo=""
        for i = lineEleEnd:lineGroEnd
            if homeDir[i,1] == "A"
                orig = homeDir[i,2]
                replaceR = homeDir[i,4]
                for j = 1:length(homeDir[:,1])
                    for k = 1:length(homeDir[1,:])
                        if homeDir[j,k] == orig
                            homeDir[j,k] = replaceR
                        end
                    end
                end
            end
        end
        for i = lineEleEnd:lineGroEnd
            if homeDir[i,1] == "T"
                tipo = homeDir[i,2]
            end
            if homeDir[i,1] == "F"
                    for j = 1:length(elems[1,:])
                        if elems[2,j] == tipo
                            r = homeDir[i,4]*homeDir[i,5]*homeDir[i,6]
                            elems[3,j] = replace(r,"**"=>"^")
                            for k = 4:2:length(elems[:,1])
                                elems[3,j] = replace(elems[3,j],elems[k,j]=>elems[k+1,j])
                            end
                        end
                    end
            end
        end
        for i = lineEleEnd:lineGroEnd
            if homeDir[i,1] == "T"
                tipo = homeDir[i,2]
            end
            if homeDir[i,1] == "R"
                z = homeDir[i,4]*homeDir[i,3]
                if homeDir[i,5] != ""
                    z = z*"+"*homeDir[i,6]*homeDir[i,5]
                end
                for j = 1:length(elems[1,:])
                    if elems[2,j] == tipo
                        elems[3,j] = replace(elems[3,j],homeDir[i,2]=>a)
                        for k = 4:2:length(elems[:,1])
                            elems[3,j] = replace(elems[3,j],elems[k,j]=>elems[k+1,j])
                        end
                    end
                end
            end
        end
    return elems
end

#CONSOLIDATE ALL GROUPS AND ELEMENTS INTO OBJECTIVE FUNCTION
function groAssig(problem::String,line::Int64,lineGroEnd::Int64,elems,gro,homeDir)
    a=""
    c=""
    b=""
    finalFunc = ""
    groHome=Array{Union{Missing, String}}(missing, 7, length(gro[1,:]))
    tipo=""
    groHome[1,:]=gro[1,:]
    groHome[2,:]=gro[2,:]
    groHome[3,:]=gro[3,:]
    groHome[4,:]=gro[4,:]
    groHome[5,:]=gro[5,:]
        for i = lineGroEnd:line
            if homeDir[i,1] == "A"
                orig = homeDir[i,2]
                replaceR = homeDir[i,4]
                for j = 1:length(homeDir[:,1])
                    for k = 1:length(homeDir[1,:])
                        if homeDir[j,k] == orig
                            homeDir[j,k] = replaceR
                        end
                    end
                end
            end
        end
        for i = lineGroEnd:line
            if homeDir[i,1] == "T"
                tipo = homeDir[i,2]
            end
            if homeDir[i,1] == "F"
                    for k = 1:length(gro[1,:])
                        if gro[3,k] == tipo
                            r = homeDir[i,4]*homeDir[i,5]*homeDir[i,6]
                            groHome[6,k] = replace(r,"**"=>"^")
                        end
                    end
            end
        end
        for i = 6:length(gro[:,1])
            for d = 2:length(gro[1,:])
                for k = 2:length(elems[1,:])
                    gro[i,d] = replace(gro[i,d],elems[1,k]=>elems[3,k])
                end
            end
        end
        for i = 1:length(groHome[:,1])
            for j = 1:length(groHome[1,:])
                if string(groHome[i,j]) == "missing"
                    groHome[i,j] = ""
                end
            end
        end
        for h = 2:length(gro[1,:])
            for d = 6:length(gro[:,1])
                if (gro[d,h]) != "" && (groHome[7,h]) != ""
                    groHome[7,h] = groHome[7,h]*" + "*gro[d,h]
                end
                if (gro[d,h]) != "" && (groHome[7,h]) == ""
                    groHome[7,h] = gro[d,h]
                end
            end
        end
        for d = 2:length(gro[1,:])
            if gro[3,d] == tipo
                groHome[7,d] = "("*groHome[7,d]*")"
                groHome[6,d] = replace(groHome[6,d],groHome[4,d]=>groHome[7,d])
                groHome[5,d] = groHome[5,d]
                if (groHome[5,d]) != ""
                    groHome[6,d] = string(groHome[5,d])*groHome[6,d]
                end
            end
        end
        for f = 2:length(gro[1,:])
            finalFunc = finalFunc*" + "*groHome[6,f]
        end
        finalFunc = chop(finalFunc,head=1,tail=0)
        for h = 2:length(elems[1,:])
            finalFunc = replace(finalFunc,elems[1,h]=>elems[3,h])
        end
        finalFunc = chop(finalFunc,head=1,tail=0)
        return finalFunc
end


prob = "OSCIGRAD.SIF"
#MATRIX OF SIF FILE
homeDir = house(prob)
# for i = 1:length(homeDir[:,1])
#     println(homeDir[i,:])
# end

#LINE SECTION BENCHMARKS
lines = counter(prob,homeDir)
line=lines[1]
lineVar=lines[2]
lineGroup=lines[3]
lineBoundz=lines[4]
lineConstz=lines[5]
lineEleType=lines[6]
lineEleUse=lines[7]
lineGroType=lines[8]
lineGroUse=lines[9]
lineObjBound=lines[10]
lineEleEnd=lines[11]
lineGroEnd=lines[12]
#NOTE THAT THIS'LL BE THE LAST LINE=LINES[1], NOT THE FIRST ENDATA
lineEnd1 = lines[13]
lineQuads = lines[14]

#PRINT LINE BENCHMARKS
# println(line)
# println(lineVar)
# println(lineGroup)
# println(lineBoundz)
# println(lineConstz)
# println(lineEleType)
# println(lineEleUse)
# println(lineGroType)
# println(lineGroUse)
# println(lineObjBound)
# println(lineEleEnd)
# println(lineGroEnd)
# println(lineEnd1)
# println(lineQuads)

#SUB IN INITIAL PARAMETERS
homeDir = initParams(prob,lineVar,line,homeDir)
# for i = 1:length(homeDir[:,1])
#     println(homeDir[i,:])
# end

gro = groupMake(prob,lineGroup,lineBoundz,lineConstz,12,homeDir)
gro = groupUse(prob,lineGroUse,lineObjBound,gro,homeDir)
gro = groupType(prob,lineGroType,lineGroUse,gro,homeDir)
gro = constAssig(prob,lineConstz,lineBoundz,gro,homeDir)

elemz = eleUseSet(prob,lineGroType,lineEleUse,10,homeDir)
elems = elemz[1]
elemTot = elemz[2]
elems = eleAssig(prob,lineEleEnd,lineGroEnd,elems,elemTot,homeDir)
finalFunc = groAssig(prob,line,lineGroEnd,elems,gro,homeDir)

println(finalFunc)