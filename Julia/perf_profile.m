function hl = perf_profile(H,gate,logplot)
%     This subroutine produces a performance profile as described in:
%
%     Benchmarking Derivative-Free Optimization Algorithms
%     Jorge J. More' and Stefan M. Wild
%     SIAM J. Optimization, Vol. 20 (1), pp.172-191, 2009.
%
%     The latest version of this subroutine is always available at
%     http://www.mcs.anl.gov/~more/dfo/
%     The authors would appreciate feedback and experiences from numerical
%     studies conducted using this subroutine.
%
%     Performance profiles were originally introduced in
%     Benchmarking optimization software with performance profiles,
%     E.D. Dolan and J.J. More', 
%     Mathematical Programming, 91 (2002), 201--213.
%
%     The subroutine returns a handle to lines in a performance profile.
%
%       H contains a three dimensional array of function values.
%         H(f,p,s) = function value # f for problem p and solver s.
%       gate is a positive constant reflecting the convergence tolerance.
%       logplot=1 is used to indicate that a log (base 2) plot is desired.
%
%     Argonne National Laboratory
%     Jorge More' and Stefan Wild. January 2008.

[nf,np,ns] = size(H); % Grab the dimensions

% Produce a suitable history array with sorted entries:
for j = 1:ns
    for i = 2:nf
      H(i,:,j) = min(H(i,:,j),H(i-1,:,j));
    end
end

prob_min = min(min(H),[],3);   % The global minimum for each problem
prob_max = H(1,:,1);           % The starting value for each problem

% For each problem and solver, determine the number of evaluations
% required to reach the cutoff value
T = zeros(np,ns);
for p = 1:np
  cutoff = prob_min(p) + gate*(prob_max(p) - prob_min(p));
  for s = 1:ns
    nfevs = find(H(:,p,s) <= cutoff,1);
    if (isempty(nfevs))
      T(p,s) = NaN;
    else
      T(p,s) = nfevs;
    end
  end
end

% Other colors, lines, and markers are easily possible:
colors  = ['b' 'r' 'k' 'm' 'c' 'g' 'y'];   lines   = {'-' '-.' '--'};
markers = [ 's' 'o' '^' 'v' 'p' '<' 'x' 'h' '+' 'd' '*' '<' ];

if (nargin < 3); logplot = 0; end

% Compute ratios and divide by smallest element in each row.
r = T./repmat(min(T,[],2),1,ns);

% Replace all NaN's with twice the max_ratio and sort.
max_ratio = max(max(r));
r(isnan(r)) = 2*max_ratio;
r = sort(r);

% Plot stair graphs with markers.
hl = zeros(ns,1);
for s = 1:ns
    [xs,ys] = stairs(r(:,s),(1:np)/np);

    % Only plot one marker at the intercept
    if (xs(1)==1)
        vv = find(xs==1,1,'last');
        xs = xs(vv:end);   ys = ys(vv:end);
    end

    sl = mod(s-1,3) + 1; sc = mod(s-1,7) + 1; sm = mod(s-1,12) + 1;
    option1 = [char(lines(sl)) colors(sc) markers(sm)];
    option2 = [char(lines(sl)) colors(sc)];
    if (logplot)
        hl(s) = semilogx(xs,ys,option2);
    else
        hl(s) = plot(xs,ys,option2);
    end
    hold on;
    % EXPERIMENTING:
    xs_all{s} = xs;
    ys_all{s} = ys;
    
    hl(s) = plot(xs(1),ys(1),option1,'MarkerFaceColor',colors(sc));
    %%%%%%%%%%%%%%%%%
end

% Process the markers
num_marks=5;
%mark_max = max(cat(1,xs_all{:}));
mark_max = max_ratio;
mark_min = min(cat(1,xs_all{:}));
if logplot
    xs_marks = logspace(log10(mark_min),log10(mark_max),10);
else
    xs_marks = mark_min:(mark_max-mark_min)/num_marks:mark_max;
end

for s = 1:ns
    x1 = xs_all{s}(1);
    xs_new = [x1; xs_marks(xs_marks >= x1)'];
    ys_new = [ys_all{s}(1); zeros(length(xs_new)-1,1)];
    for i = 2:length(xs_new)
        j = find(xs_all{s} >= xs_new(i),1,'first');
        if isempty(j)
            ys_new(i) = 1;
        else
            ys_new(i) = ys_all{s}(j);
        end
    end    
    
    sc = mod(s-1,7) + 1; sm = mod(s-1,12) + 1;      
    option3 = [colors(sc) markers(sm)];

    scatter(xs_new,ys_new,option3,'MarkerFaceColor',colors(sc),'SizeData',30); % Just plot markers       
end

% Axis properties are set so that failures are not shown, but with the
% max_ratio data points shown. This highlights the "flatline" effect.
if (logplot) 
  axis([1 1.1*max_ratio 0 1]);
  twop = floor(log2(1.1*max_ratio));
  set(gca,'XTick',2.^[0:twop])
else
  axis([1 1.1*max_ratio 0 1]);
end
