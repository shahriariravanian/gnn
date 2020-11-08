using DifferentialEquations
using Random: shuffle
using Dates: now


function patched_state(v, nv, p, tspan = (1000.0, 5000.0); dtmax=0.25, Δt=1.0)
    f = (du,u,p,t) -> begin
        u[nv] = v
        f!(du, u, p, t)
    end

    u = copy(u0)
    prob = ODEProblem(f, u, tspan, p)
    sol = solve(prob, TRBDF2(), dtmax=dtmax, saveat=Δt)
    u = sol.u[end]
    u[nv] = v
    return u
end

function voltage_clamp(T, ℳ, V=-100.5:1.0:50.5; dtmax=0.25, Δt=1.0)
    K = zeros(T, (length(u0), length(V)))

    for (i,v) in enumerate(V)
        K[:,i] = patched_state(v, ℳ.nv, ℳ.p; dtmax=dtmax, Δt=Δt)
    end

    return K
end

function simulate(T, ℳ; cls=300:5:800, tspan=(0.0,20000.0), Δt=1.0, dtmax=0.25)
    prob = ODEProblem(ℳ.f!, ℳ.u0, tspan, ℳ.p)

    U = Array{T,2}[]
    q = copy(ℳ.p)

    for cl in cls
        q[ℳ.icl] = cl
        sol = solve(prob, TRBDF2(), dtmax=dtmax, saveat=Δt, p=q)
        u = Array(sol)
        push!(U, u)
    end

    return U
end

function create_normalizer(U, K)
    lo = minimum(hcat([minimum(u, dims=2) for u in U]...), dims=2)
    hi = maximum(hcat([maximum(u, dims=2) for u in U]...), dims=2)

    lo = min.(lo, minimum(K, dims=2))
    hi = max.(hi, maximum(K, dims=2))

    return x -> (x .- lo) ./ (hi .- lo)
end

function prepare_seq_data(U::Array{T}, span, normalizer) where {T}
    ns = size(U[1], 1)
    m = length(U)
    r = shuffle(1:m)
    # r = 1:m

    X = T[]

    for t = span[1]:span[2]
        x = zeros(eltype(T), (ns, m))

        for i in r
            k = r[i]
            x[:,i] = normalizer(U[k][:,t])
        end
        push!(X, x)
    end

    return X
end

function mix_dynamic_static(X, K, normalizer, nk; mix=false)
    r = shuffle(1:size(K,2))[1:nk]
    k = normalizer(K[:,r])
    return [[x k] for x in X]
end

function prepare(ℳ1, ℳ2; mix=false)
    @info "calculating U0: $(now())"
    U0 = simulate(Float32, ℳ1)
    K = voltage_clamp(Float32, ℳ1)
    normalizer = create_normalizer(U0, K)

    @info "calculating U2: $(now())"
    U2 = simulate(Float32, ℳ2)

    @info "calculating Xs: $(now())"
    X0 = prepare_seq_data(U0, (9000,19000), normalizer)
    X2 = prepare_seq_data(U2, (9000,19000), normalizer)

    if mix
        X0 = mix_dynamic_static(X0, K, normalizer, 50)
        X2 = mix_dynamic_static(X2, K, normalizer, 50)
    end

    return NamedTuple{(:U0, :U2, :X0, :X2, :mask, :obs)}(
                      ( U0,  U2,  X0,  X2, ℳ1.mask, ℳ1.obs))
end
