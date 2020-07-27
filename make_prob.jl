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
    S = Matrix(F.Q)
    λs = rand(c+1:2c-1,n)
    λs[1]=c
    λs[n]=2c
    Λ = lambda_make(shuffle(λs))
    #println(Λ)
    Q = S*Λ*S'
    return Q
end

function prob2(n,c=1)
    A = rand(n,n)
    F = qr(A)
    S = Matrix(F.Q)
    λs = rand(c+1:20c-1,n)
    λs[1]=c
    λs[n]=20c
    Λ = lambda_make(shuffle(λs))
    #println(Λ)
    Q = S*Λ*S'
    return Q
end

function prob3(n,c=5)
    A = rand(n,n)
    F = qr(A)
    S = Matrix(F.Q)
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

function prob4(n)
    
end
#Q,f,∇f = prob_function(n,prob2)
