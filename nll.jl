include("flow.jl")


function nll(theta)
    
    # ptilde0 interpolation based on theta
    ptilde0 = ode_interpolate(theta)
    
    # initialize log likelihood
    yLy = 0
    logdet = 0
    # For each point in the dataset
   
    for i in 1:n
       
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

    0.5* ( n * log(2*pi) + logdet + yLy)
end


