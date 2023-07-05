using Random
using Optim
using NearestNeighbors
using Vecchia
using StaticArrays
using Distributions

include("flow.jl")

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

n0 = 5
points = random_points_in_circle(n0)
locs = [ [points[i][1],points[i][2], xi[j]]  for i in 1:n0 for j in 1:h]
n = length(locs)

m = 5

# Preallocate a list to store the indices of the nearest neighbors for each point
m_nearest_neighbors_indices = []

for i in 1:n
    # Subset the locations to only include current and previous points
    current_points = locs[1:i]
    kdtree = KDTree(reshape(hcat(current_points...), 3, :)) 

    # Find the m nearest neighbors for current point among the previous points
    idxs, _ = knn(kdtree, locs[i], min(i, m), true)

    # Store the indices of neighbors instead of the neighbors themselves
    neighbor_indices = sort([1:i...][idxs])

    push!(m_nearest_neighbors_indices, neighbor_indices)
end

function log_likelihood_vecchia(theta, y, m_nearest_neighbors_indices)
    
    # ptilde0 interpolation based on theta
    ptilde0 = ode_interpolate(theta)
    
    # initialize log likelihood
    log_likelihood = 0.0
    yLy = 0
    logdet = 0
    # For each point in the dataset
    for i in 1:n
        # Get the i-th location and its m nearest neighbors indices
        loc = locs[i]
        neighbor_indices = m_nearest_neighbors_indices[i]
        K(xt1, xt2) = K0(ptilde0(xt1[1:2], xt1[3]), ptilde0(xt2[1:2], xt2[3])) + (xt1==xt2)

        # Define the covariance matrix
        cov_matrix = [K(locs[j1], locs[j2]) for j1 in neighbor_indices, j2 in neighbor_indices]

        # Compute the Cholesky decomposition of the covariance matrix
        chol_factor = cholesky(cov_matrix).L
        
        # Solve the linear system
        Ly = chol_factor \ y[neighbor_indices]
        yLy += ((Ly[end])^2)
        logdet += 2*log(chol_factor[end,end]) 

        # Accumulate the log likelihood contribution of the i-th point
    end

    return  -0.5* ( n * log(2*pi) + logdet + yLy)
end

#cov(xt1, xt2, theta) = K0(p0(xt1[1:2], xt1[3],theta), p0(xt2[1:2], xt2[3],theta)) + (xt1==xt2)

y = rand(n)
theta = rand(2*l)
Sigma = [cov(xt1,xt2,theta) for xt1 in locs, xt2 in locs]

mvn = MultivariateNormal(zeros(n), Sigma)

println(log_likelihood_vecchia(theta,y,m_nearest_neighbors_indices))
log(pdf(mvn,y))
