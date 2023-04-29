#using Pkg

#dependencies = [
#    "Distributions",
#    "QuadGK",
#    "Optim",
#    "Random",
#    "ForwardDiff",
#    "Flux"
#]

#Pkg.add(dependencies)
 
using Distributions, Flux, QuadGK, Optim, Random, ForwardDiff

 
pdf(x, a) = a * exp(-a * x)

function log_likelihood(params, x)
	a  = params
	n = length(x)
	return sum(log.(pdf.(x, a)))
end

function neg_log_likelihood(params, x)
	-log_likelihood(params, x)
end

function cdf(x, a)
	result, _ = quadgk(t -> pdf(t, a), 0, x)
	result
end

function inv_cdf(y, a)
	result = optimize(x -> (cdf(x, a) - y)^2, -10.0, 10.0)
	result.minimizer
end

a= 1.12212
n= 1000
u = rand(n)
x = inv_cdf.(u, Ref(a))


# Optimization
params0 = [a]
func = TwiceDifferentiable(vars -> neg_log_likelihood(vars, x), params0; autodiff=:forward)
opt = optimize(func, params0)

# Get MLE estimates
a_mle  = Optim.minimizer(opt)
println("MLE estimates: a = $a_mle") 

 

function regression_model(input_dim, hidden_dim, output_dim)
    return Chain(
        Dense(input_dim, hidden_dim, relu),
        Dense(hidden_dim, output_dim)
    )
end

train_data = rand(10, 5)  # 10 training examples, 5 features
train_labels = rand(10, 1)  # 10 training labels, 1 output
test_data = rand(5, 5)  # 5 test examples, 5 features
test_labels = rand(5, 1)  # 5 test labels, 1 output

model = regression_model(5, 10, 1)

loss(x, y) = Flux.mse(model(x), y)

optimizer = ADAM()
batch_size = 5
num_epochs = 100
for epoch in 1:num_epochs
    for (x, y) in zip(collect(eachrow(Float32.(train_data))), collect(eachrow(Float32.(train_labels))))
        Flux.train!(loss, Flux.params(model), [(x, y)], optimizer)
    end
end

test_predictions = model(Float32.(test_data))

println("Test theta_hat: ", test_predictions)
println("Test theta: ", test_labels)











