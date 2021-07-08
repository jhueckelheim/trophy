function power(x::Vector)
  grad = zeros(size(x))
  sum = 0
  for i = 1:length(x)
    term = i*x[i]^2
    sum = sum + term
  end
  for i = 1:length(x)
    grad[i] = 2*sum*2*i*x[i]
  end
  sum = sum^2
  return sum, grad
end
