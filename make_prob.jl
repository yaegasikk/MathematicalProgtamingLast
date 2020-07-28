using Random,LinearAlgebra
Random.seed!(7239)

function lambda_make(λs)
    n = size(λs)[1]
    Λ = zeros(n,n)
    for i in 1:n
        Λ[i,i]=λs[i]
    end
    return Λ   
end

function prob1(n,c=10)
    A = rand(n,n)
    F = qr(A)
    S = F.Q
    λs = rand(c+1:2c-1,n)
    λs[1]=c
    λs[n]=2c
    Λ = lambda_make(shuffle(λs))
    #println(Λ)
    Q = S*Λ*S'
    return Q
end

function prob2(n,c=10)
    A = rand(n,n)
    F = qr(A)
    S = F.Q
    λs = rand(c+1:20c-1,n)
    λs[1]=c
    λs[n]=20c
    Λ = lambda_make(shuffle(λs))
    #println(Λ)
    Q = S*Λ*S'
    return Q
end

function prob3(n,c=3)
    A = rand(n,n)
    F = qr(A)
    S = F.Q
    l = ceil(Int64,0.3*n)
    λs = rand(c:2c,n)
    for i in 1:l
        λs[i] = 0
    end
    Λ = lambda_make(shuffle(λs))
    #println(Λ)
    Q = S*Λ*S'
    return Q
end

function prob_function(n,probq,c=0)
    xs = rand(n)
    if c==0
        Q = probq(n)
    else
        Q = probq(n,c)
    end
    f(x) = (1/2).*(((x-xs)')*Q*(x-xs))
    ∇f(x) = Q*(x-xs)
    return Q, f, ∇f,xs    
end

function prob4(x)
    n = size(x)[1]
    return_ans = 0.0
    for i in 2:n
        return_ans = return_ans + 100*(x[i]-x[i-1]^2)^2 + (1-x[i])^2
    end
    return return_ans
end

function nablaprob4(x)
    n = size(x)[1]
    return_list = zeros(size(x))
    return_list[1] = -400*x[1]*(x[2]-x[1]^2)
    for i in 2:n-1
        return_list[i] = 200*(x[i]-x[i-1]^2)-(2*(1-x[i]))-(400*(x[i]*(x[i+1]-x[i]^2)))
    end
    return_list[n] = 200*(x[n]-x[n-1]^2)-(2*(1-x[n]))
    return return_list
end

#Q,f,∇f = prob_function(n,prob2)
