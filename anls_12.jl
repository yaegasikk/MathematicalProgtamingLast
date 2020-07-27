include("utils.jl")
include("make_prob.jl")
using Random,LinearAlgebra

seed = 72331
stop =1.0e-8
function steepestdescent(n,prob)
    Random.seed!(seed)
    Q,f,∇f,xs = prob_function(n,prob)    
    x = rand(n)
    stop_list = [stop_criterion(f,x,xs)]
    while stop_criterion(f,x,xs)>stop
        d = (-1)*∇f(x)
        α = lineserch_pd(x,Q,∇f)
        #α = lineserch_back(x,f,∇f)
        x = x .+ α.*d
        push!(stop_list,stop_criterion(f,x,xs))
        println(stop_criterion(f,x,xs))
    end

    return stop_list
end

function conjugategradient(n,prob)
    Random.seed!(seed)
    Q,f,∇f,xs = prob_function(n,prob)    
    x_k = rand(n)
    stop_list = [stop_criterion(f,x_k,xs)]
    p = (-1)*∇f(x_k)
    while stop_criterion(f,x_k,xs)>stop
        α = lineserch_pd(x_k,Q,∇f)
        #α = lineserch_back(x_k,f,∇f)
        x_k1 = x_k + α.*p
        p = (-1).*∇f(x_k1)+((∇f(x_k1)'*Q*p)/(p'*Q*p)).*p
        push!(stop_list,stop_criterion(f,x_k1,xs))
        println(stop_list[end])
        x_k = x_k1   
    end
    return stop_list
end

function qnewtonBFGS(n,prob)
    Random.seed!(seed)
    Q,f,∇f,xs = prob_function(n,prob)    
    x_k = rand(n)
    stop_list = [stop_criterion(f,x_k,xs)]
    H = Matrix{Float64}(I, n, n)
    d = (-1).*H*∇f(x_k)
    α = lineserch_pd(x_k,Q,∇f)
    #α = lineserch_back(x_k,f,∇f)
    x_k1 = x_k + α.*d
    s = x_k1 - x_k
    y = ∇f(x_k1).-∇f(x_k)
    b = ((s*(H*y)')+(H*y*s'))/(s'*y)
    c = (1+((y'*H*y)/(s'*y)))*((s*s')/(s'*y))
    H = H.-b.+c
    x_k = x_k1
    push!(stop_list,stop_criterion(f,x_k1,xs))
    while stop_criterion(f,x_k,xs)>stop
        d = (-1).*H*∇f(x_k)
        α = lineserch_pd(x_k,Q,∇f)
        #α = lineserch_back(x_k,f,∇f)
        x_k1 = x_k + α.*d
        s = x_k1 - x_k
        y = ∇f(x_k1)-∇f(x_k)
        b = ((s*(H*y)')+(H*y*s'))/(s'*y)
        c = (1+((y'*H*y)/(s'*y)))*((s*s')/(s'*y))
        H = H.-b.+c
        x_k = x_k1
        push!(stop_list,stop_criterion(f,x_k1,xs))
        #println(stop_list[end])
    end
    return stop_list    
end

gr()
a = conjugategradient(100,prob1)
plot(a,yaxis=:log,label="CG",ylabel="|f(xk)-f(x*)|",xlabel="k")
println(size(a))
a = steepestdescent(100,prob1)
println(size(a))
plot!(a,yaxis=:log,label="SD")
a = qnewtonBFGS(100,prob1)
println(size(a))
plot!(a,yaxis=:log,label="BFGS")
savefig("ans1.pdf")

