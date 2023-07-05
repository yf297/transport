using Random
using Optim
using Distributions

include("nn.jl")
include("nll.jl")

K0(x1, x2) = exp(-0.5*norm(x1 - x2)^2)

function random_points_in_circle(n)
    points = []
    for _ in 1:n
        angle = 2π*rand()  # Random angle
        radius = r*sqrt(rand())  # Random radius, use sqrt for uniform distribution in the circle
        x = radius*cos(angle)  # Convert polar to cartesian coordinates
        y = radius*sin(angle)
        push!(points, [x, y])
    end
    return points
end

n0 = 100
points = random_points_in_circle(n0)
locs = [ [points[i][1],points[i][2], xi[j]]  for i in 1:n0 for j in 1:h]
n = length(locs)

m = 30

m_nearest_neighbors_indices = find_nn(locs, m)

#cov(xt1, xt2, theta) = K0(p0(xt1[1:2], xt1[3],theta), p0(xt2[1:2], xt2[3],theta)) + (xt1==xt2)

y = rand(n)
theta = rand(2*l)
#Sigma = [cov(xt1,xt2,theta) for xt1 in locs, xt2 in locs]

#mvn = MultivariateNormal(zeros(n), Sigma)

println(nll(theta))
#log(pdf(mvn,y))
