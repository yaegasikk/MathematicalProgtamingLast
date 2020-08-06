include("utils.jl")
include("make_prob.jl")
using Random,LinearAlgebra

seed = 72330
stop =1.0e-3

function steepestdescent(n)
    Random.seed!(seed)
    f(x) = prob4(x)
    ∇f(x) = nablaprob4(x)
    xs = ones(Float64,n)
    x = rand(n).*10
    stop_list = [stop_criterion(prob4,x,xs)]
    while stop_list[end]>=stop
        d = (-1)*∇f(x)
        #α = lineserch_pd(x,Q,∇f)
        α = lineserch_back(x,f,∇f)
        x = x + (α.*d)
        println(α)
        push!(stop_list,stop_criterion(f,x,xs))
        #println(stop_criterion(f,x,xs))
    end

    return stop_list
end

function qnewtonBFGS(n)
    Random.seed!(seed)
    f(x) = prob4(x)
    ∇f(x) = nablaprob4(x)
    xs = ones(Float64,n)  
    x_k = rand(n).*10
    stop_list = [stop_criterion(f,x_k,xs)]
    H = Matrix{Float64}(I, n, n)
    d = (-1).*H*∇f(x_k)
    α = lineserch_back(x_k,f,∇f)
    x_k1 = x_k + α.*d
    s = x_k1 - x_k
    y = ∇f(x_k1)-∇f(x_k)
    b = ((H*y*s')+(s*y'*H'))/(y'*s)
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
        b = ((H*y*s')+(s*y'*H'))/(y'*s)
        c = (1+((y'*H*y)/(s'*y)))*((s*s')/(s'*y))
        H = H.-b.+c
        x_k = x_k1
        push!(stop_list,stop_criterion(f,x_k1,xs))
        println(stop_list[end])
    end
    return stop_list    
end

function nonlinerFR(n)
    Random.seed!(seed)
    f(x) = prob4(x)
    ∇f(x) = nablaprob4(x)
    xs = ones(Float64,n)  
    x_k = rand(n).*10
    p = (-1)*∇f(x_k)
    stop_list = [stop_criterion(f,x_k,xs)]
    println(stop)
    while stop_list[end]>stop
        α = lineserch_back(x_k,f,∇f)
        x_k1 = x_k + α.*p
        push!(stop_list,stop_criterion(f,x_k1,xs))
        println(stop_list[end])
        β = (norm(∇f(x_k1))^2)/(norm(∇f(x_k))^2)
        p = (-1).*∇f(x_k1)+β.*p
        x_k = x_k1
    end
    return stop_list
end

function nonlinerPR(n)
    Random.seed!(seed)
    f(x) = prob4(x)
    ∇f(x) = nablaprob4(x)
    xs = ones(Float64,n)  
    x_k = rand(n).*10
    p = (-1)*∇f(x_k)
    stop_list = [stop_criterion(f,x_k,xs)]
    println(stop)
    while stop_list[end]>stop
        α = lineserch_back(x_k,f,∇f)
        x_k1 = x_k + α.*p
        push!(stop_list,stop_criterion(f,x_k1,xs))
        println(stop_list[end])
        y = ∇f(x_k1)-∇f(x_k)
        β = (∇f(x_k1)'*y)/(p'*y)
        p = (-1).*∇f(x_k1)+β.*p
        x_k = x_k1
    end
    return stop_list
end

a = steepestdescent(1000)
println(size(a))

