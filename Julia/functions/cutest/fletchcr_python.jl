function fletchcr(x::Vector)
  #println("Julia port of CUTEST's FLETCHCR")
  T = typeof(x[1])
  #grad = zeros(size(x))
  grad = zeros(T, size(x))
  sum = 0
  for i = 1:(length(x)-1)
    term1 = -x[i]^2 + x[i+1]
    term2 = -x[i]+1
    sum = sum + 100*term1^2 + term2^2
    grad[i] = grad[i] + 2*100*term1*(-2)*x[i] + 2*term2*(-1)
    grad[i+1] = grad[i+1] + 2*100*term1
  end
  return sum, grad
end
