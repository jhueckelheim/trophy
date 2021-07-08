
function oscigrad(x::AbstractVector)
  println("Julia port of CUTEST's FLETCHCR")  # is this correct
  #grad = zeros(size(x))
  sum = 0.5*x[1]-0.5-4*500*(x[2]-2*x[1]^2+1)*x[1]
  sum = sum^2
  for i = 2:(length(x)-1)
    term1 = -4*500*(x[i+1]-2*x[i]^2+1)*x[i]
    term2 = 2*500*(x[i]-2*x[i-1]^2+1)
    sum = sum + (term1 + term2)^2
  end
  term2 = 2*500*(x[length(x)]-2*x[length(x)-1]^2+1)
  sum = sum + term2^2
  return sum#, grad    NO GRADIENT
end


#y = ones(100000)
#A = oscigrad(y)
#B = wrapfun(y,"OSCIGRAD")
#end