using Flux
using Flux: @epochs, onehotbatch, mse, mae, throttle, update!
using CUDAapi
using CUDAnative
using Statistics: mean
using Printf: @printf

if has_cuda()
    @info "CUDA is on"
    import CuArrays
    CuArrays.allowscalar(false)
end

find_gnn(m) = findfirst(map(x -> x isa Flux.Recur, m))

mse2(ŷ, y, nt) = begin
    x = (ŷ .- y).^2
    return mean(x[:,1:nt]), mean(x[:,nt+1:end])
end

mae2(ŷ, y, nt) = begin
    x = abs.(ŷ .- y)
    return mean(x[:,1:nt]), mean(x[:,nt+1:end])
end

function train_base_model(X, mask, obs; nh=30, epochs=3, λ::Float32=1f-4, ρ=0.75)
    obs, mask = cpu(obs), cpu(mask)
    nobs = length(obs)
    ns = size(X[1], 1)
    nt = round(Int, size(X[1], 2) * ρ)
    gates = (mask .== 0)
    ng = sum(gates)
    nc = sum(.~gates)

    O = gpu([x[obs,:] for x in @view(X[1:end-1])])
    G = gpu([x[gates,:] for x in @view(X[2:end])])
    C = gpu([x[.~gates,:] for x in @view(X[2:end])])

    m1 = Chain(
        Dense(nobs, nh, relu),
        Dense(nh, nh, relu),
        GNN(nh, ng)
    )

    m2 = Chain(
        Dense(nobs+ng, nh, relu),
        Dense(nh, nh, relu),
        LSTM(nh, nc),
        Dense(nc, nc, σ)
    )

    m1 = gpu(m1)
    m2 = gpu(m2)

    p1 = params(m1)
    p2 = params(m2)

    p = params(m1, m2)

    l = 0.0
    l1 = 0.0
    l2 = 0.0
    lv = 0.0
    l1v = 0.0
    l2v = 0.0

    cb = throttle(1.0) do
        @printf "l1 = %.5f\tl2 = %.5f\tl1v = %.5f\tl2v = %.5f\r" l1 l2 l1v l2v
    end

    opt = ADAM(0.0001)

    fnorm(x) = sum(abs2, x) / length(x)

    loss(x, g, c) = begin
        h = m1(x)
        l1, l1v = mse2(h, g, nt)
        l2, l2v = mse2(m2([x; h]), c, nt)
        r = λ * sum(fnorm, p)
        l = l1 + l2 + r
        lv = l1v + l2v + r
        return l
    end

    @epochs epochs Flux.train!(loss, p, zip(O,G,C), opt; cb=cb)

    println()
    println()
    return m1, m2
end

function retrain_model(m1, m2, X, mask, obs; epochs=5, λ::Float32=1f-5, η=1f0, ρ=0.75)
    obs, mask = cpu(obs), cpu(mask)
    Ω = cumsum(mask)[obs]
    nobs = length(obs)
    ns = size(X[1], 1)
    nt = round(Int, size(X[1], 2) * ρ)
    gates = (mask .== 0)
    ng = sum(gates)
    nc = sum(.~gates)

    O = [x[obs,:] for x in X]
    O1 = gpu(O[1:end-1])
    O2 = gpu(O[2:end])

    Flux.reset!(m1)
    Flux.reset!(m2)

    # m = gpu(deepcopy(m1))
    m = gpu(Chain(m1[1:end-1]..., deepcopy(m1[end])))

    fnorm(x) = sum(abs2, x) / length(x)

    l = 0.0
    l1 = 0.0
    l2 = 0.0
    l1v = 0.0
    l2v = 0.0

    cb = throttle(1.0) do
        print(repeat(' ', 75), '\r')
        @printf "l1 = %.5f\tl2 = %.5f\tl1v = %.5f\tl2v = %.5f\r" l1 l2 l1v l2v
    end

    opt = ADAM(0.0001)

    p = params(m[end])

    loss(x, y) = begin
        h = m(x)
        l1, l1v = mae2(h, m1(x), nt)
        w = m2([x; h])
        l2, l2v = mse2(w[Ω,:], y, nt)
        r = λ * sum(fnorm, p)
        l = η * l1 + l2 + r
        lv = η * l1v + l2v + r
        return l
    end

    @epochs epochs Flux.train!(loss, p, zip(O1,O2), opt; cb=cb)

    println()
    println()
    return m
end

function train(ℬ)
    m1, m2 = train_base_model(ℬ.X0, ℬ.mask, ℬ.obs; epochs=3)
    return merge(ℬ, (m1=m1, m2=m2))
end

function retrain(ℬ, η)
    mp = retrain_model(ℬ.m1, ℬ.m2, ℬ.X0, ℬ.mask, ℬ.obs; epochs=3, η=η)
    mq = retrain_model(ℬ.m1, ℬ.m2, ℬ.X2, ℬ.mask, ℬ.obs; epochs=3, η=η)
    return (mp=mp, mq=mq, η=η)
end
