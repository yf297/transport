using Distributions

include("flow.jl")


function nll(theta)
    p = ode_interpolate(theta)
    K(x1, t1, x2, t2) = K0(p(x1, t1), p(x2, t2)) + min(t1, t2)
    
    n = length(y)
    Sigma = zeros(n, n)  # initialize Sigma as a zero matrix of size n x n
    for i in 1:n
        for j in 1:n
            x_i, t_i = locs[i]
            x_j, t_j = locs[j]
            Sigma[i, j] = K(x_i, t_i, x_j, t_j)
        end
    end
   
    Sigma = Sigma + 0.01*I
    m = zeros(n)
    dist = MvNormal(m, Sigma)
    return -logpdf(dist, y)
end    





