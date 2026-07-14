function some_comp(n::Int64)
  return summ(sin(i)*cos(i) for i in 1:n)
end

some_comp(100000000)
