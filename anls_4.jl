include("utils.jl")
include("make_prob.jl")
using Random,LinearAlgebra

seed = 72331
stop =1.0e-2

function steepestdescent(n)
    Random.seed!(seed)
    f(x) = prob4(x)
    ∇f(x) = nablaprob4(x)
    xs = ones(Float64,n)
    x = rand(n)
    println(f(xs))
    stop_list = [stop_criterion(f,x,xs)]
    while stop_criterion(f,x,xs)>stop
        d = (-1)*∇f(x)
        #α = lineserch_pd(x,Q,∇f)
        α = lineserch_back(x,f,∇f)
        x = x .+ α.*d
        push!(stop_list,stop_criterion(f,x,xs))
        println(stop_criterion(f,x,xs))
    end

    return stop_list
end

a = steepestdescent(10)
println(size(a))
