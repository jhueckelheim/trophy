function [handle_data,handle_perf] = generate_profiles(gate,logplot,plottype)

fs = 16;

H = NaN*ones(4*550,100,3);
N = NaN*ones(100,1); 

index = 0;
for prob = 1:103
    if prob ~= 14 && prob ~= 82 && prob ~= 85
        index = index + 1;
        
        % double
        probdata = strcat('dprob',num2str(prob),'.mat');
        load(probdata)
        fprobdata = strcat('f',probdata);
        load(fprobdata)
        num_evals = length(fvals); 
        for k = 1:num_evals
            if strcmp(plottype,'f')
                H(4*(k-1)+1:4*k,index,1) = repmat(fvals(k),1,4);
            else
                H(4*(k-1)+1:4*k,index,1) = repmat(normgvals(k),1,4);
            end
        end   
        
        % single
        probdata = strcat('sprob',num2str(prob),'.mat');
        load(probdata)
        fprobdata = strcat('f',probdata);
        load(fprobdata)
        num_evals = length(fvals); 
        for k = 1:num_evals
            if strcmp(plottype,'f')
                H(k,index,2) = fvals(k);
            else
                H(k,index,2) = normgvals(k);
            end
        end 
        
        % dynamic
        probdata = strcat('prob',num2str(prob),'.mat');
        load(probdata)
        fprobdata = strcat('f',probdata);
        load(fprobdata)
        num_evals = length(fvals); 
        ctr = 1;
        for k = 1:num_evals
            if strcmp(plottype,'f')
                if strcmp(prechist(k+1),'s')
                    H(ctr,index,3) = fvals(k);
                    ctr = ctr + 1;
                elseif strcmp(prechist(k+1),'sdd')
                    H(ctr:(ctr+8),index,3) = repmat(fvals(k),1,9);
                    ctr = ctr + 9;
                elseif strcmp(prechist(k+1),'d')
                    H(ctr:(ctr+3),index,3) = repmat(fvals(k),1,4);
                    ctr = ctr + 4;
                end
            else
                if strcmp(prechist(k+1),'s')
                    H(ctr,index,3) = normgvals(k);
                    ctr = ctr + 1;
                elseif strcmp(prechist(k+1),'sdd')
                    H(ctr:(ctr+8),index,3) = repmat(normgvals(k),1,9);
                    ctr = ctr + 9;
                elseif strcmp(prechist(k+1),'d')
                    H(ctr:(ctr+3),index,3) = repmat(normgvals(k),1,4);
                    ctr = ctr + 4;
                end
            end
        end 
        N(index) = size(xhist,2);
    end
end

figure;
handle_data=data_profile(H,N,gate,logplot);
leg_data = legend(handle_data,'double-TR','single-TR','TRUMP','Location','SouthEast');
set(leg_data,'FontSize',fs)
if strcmp(plottype,'f')
    titlestr = strcat('Data profile, $f(x)$, $\sigma = $',num2str(gate));
else
    titlestr = strcat('Data profile, $\|\nabla f(x)\|$, $\sigma = $',num2str(gate));
end
title(titlestr,'interpreter','latex','FontSize',fs);

figure;
handle_perf=perf_profile(H,gate,logplot);
leg_perf = legend(handle_perf,'double-TR','single-TR','TRUMP','Location','SouthEast');
set(leg_perf,'FontSize',fs)
if strcmp(plottype,'f')
    titlestr = strcat('Performance profile, $f(x)$, $\sigma = $',num2str(gate));
else
    titlestr = strcat('Performance profile, $\|\nabla f(x)\|$, $\sigma = $',num2str(gate));
end
title(titlestr,'interpreter','latex','FontSize',fs);

end