using Random,LinearAlgebra
#using Plots

seed =4000 

function lineserch_pd(x,Q,∇f)
    g = (-1).*∇f(x)
    alpha = (g'*g)/(g'*Q*g)
    return alpha
end

function lineserch_back(x,f,∇f,c1=0.9)
    Random.seed!(seed)
    ρ = rand()
    alpha = 1
    d = (-1).*∇f(x)
    phi(a) = f(x.+(a.*d))
    phidash(a) = d'* ∇f(x.+(a.*d))
    a = phi(alpha)
    b = (phi(0)+c1*phidash(0)*alpha)
    x = a>b
    while x
        alpha = alpha*ρ
        a = phi(alpha)
        b = phi(0.0)+c1*phidash(0.0)*alpha
        x = a>b
    end
    return alpha
end

function stop_criterion(f,x,xs)
    return abs(f(x)-f(xs))
end


