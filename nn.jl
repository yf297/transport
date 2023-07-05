using NearestNeighbors



function find_nn(locs, m)
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
    m_nearest_neighbors_indices
end

