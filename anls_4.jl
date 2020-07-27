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

function qnewtonBFGS(n)
    Random.seed!(seed)
    f(x) = prob4(x)
    ∇f(x) = nablaprob4(x)
    xs = ones(Float64,n)  
    x_k = rand(n)
    stop_list = [stop_criterion(f,x_k,xs)]
    H = Matrix{Float64}(I, n, n)
    d = (-1).*H*∇f(x_k)
    α = lineserch_back(x_k,f,∇f)
    x_k1 = x_k + α.*d
    s = x_k1 - x_k
    y = ∇f(x_k1)-∇f(x_k)
    b = ((s*(H*y)')+(H*y*s'))/(s'*y)
    c = (1+((y'*H*y)/(s'*y)))*((s*s')/(s'*y))
    H = H.-b.+c
    x_k = x_k1
    push!(stop_list,stop_criterion(f,x_k1,xs))
    while stop_criterion(f,x_k,xs)>stop
        d = (-1).*H*∇f(x_k)
        α = lineserch_back(x_k,f,∇f)
        x_k1 = x_k + α.*d
        s = x_k1 - x_k
        y = ∇f(x_k1)-∇f(x_k)
        b = ((s*(H*y)')+(H*y*s'))/(s'*y)
        c = (1+((y'*H*y)/(s'*y)))*((s*s')/(s'*y))
        H = H.-b.+c
        x_k = x_k1
        push!(stop_list,stop_criterion(f,x_k1,xs))
        println(stop_list[end])
    end
    return stop_list    
end

a = qnewtonBFGS(100)
println(size(a))

