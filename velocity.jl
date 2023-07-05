using LinearAlgebra
using ForwardDiff


# Spatial domain is circle centered at origin with radius r. time domain is [0,T]
r = 3
T = 5
a = 2.5

# h defines the dimension of the grid. There will be l = h*h centers and 2*l parameters 
h = 2 
l = h*h

# Define the centers as grid on the largest square fitting in the circle
c1 = range(-r*sqrt(2), r*sqrt(2), length = h)
c2 = range(-r*sqrt(2), r*sqrt(2), length = h)
c = [[c1[i], c2[j]] for i in 1:h for j in 1:h]

# Discretize time with h time points
xi = range(0, T, length = h)

# b is bump function which is 1 on the circle with radius a and then decays to 0 at r
f(x) = (x > 0) * exp(-1/x)
g(x) = f(x)/(f(x) + f(1-x))
b(x) = 1 - g( (norm(x) - a^2)/(r^2 - a^2) )


# basis 
varphi(x, ci) = b(x) * exp(norm(x - ci))

# vector of the gradients and curls
function grad_curl_varphi(x) 
    grad = [ForwardDiff.gradient(x -> varphi(x, c[i]), x) for i in 1:l]
    curl = [[-grad[i][2], grad[i][1]] for i in 1:l]
    grad, curl
end

# velocity field
function v(x, theta, t)
    theta1 = theta[1:l]
    theta2 = theta[(l+1):(2*l)]
    grad, curl = grad_curl_varphi(x)
    sum(theta1[i] * grad[i] for i in 1:l) + sum(theta2[i] * curl[i] for i in 1:l)
end







