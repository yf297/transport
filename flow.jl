using DifferentialEquations
using Interpolations

include("velocity.jl")

function p0(x,t,theta)
    prob = ODEProblem(v, x, (t, 0), theta)
    sol = solve(prob)
    sol[end]
end


function ode_interpolate(theta)
    grid = zeros(h, h, h)
    p_dim1 = similar(grid)
    p_dim2 = similar(grid)

    Threads.@threads for ind in 1:(h*h*h)
        i, j, k = Tuple(CartesianIndices((h, h, h))[ind])
        x = [c1[i], c2[j]]
        t = xi[k]
        sol = p0(x, t, theta)
        p_dim1[i, j, k] = sol[1]
        p_dim2[i, j, k] = sol[2]
    end
    
    itp_dim1 = CubicSplineInterpolation((c1, c2, xi), p_dim1)
    itp_dim2 = CubicSplineInterpolation((c1, c2, xi), p_dim2)
    return (x, t) -> [itp_dim1[x[1], x[2], t], itp_dim2[x[1], x[2], t]]
end



