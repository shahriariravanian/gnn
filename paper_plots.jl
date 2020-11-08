using Plots
using Flux

include("model.jl")
include("gnn.jl")
include("prepare.jl")
include("neural_ode.jl")

const models_dir="models2"

const ℳ = TenTusscherModel()
const obs = ℳ.obs
const nv = ℳ.nv

if ! @isdefined ℬl
    @info "loading longqt"
    const ℬl = load_model(model_path(models_dir, "longqt"))
    @info "calculating normalizer"
    const K = voltage_clamp(Float32, ℳ)
    const normalizer = create_normalizer(ℬl.U0, K)
end

if ! @isdefined ℬs
    @info "loading shortqt"
    const ℬs = load_model(model_path(models_dir, "shortqt"))
end

if ! @isdefined ℬt
    @info "loading ito"
    const ℬt = load_model(model_path(models_dir, "ito"))
end


# Figure
function plot_signal_base()
    p1 = plot(ℬl.U0[101][4,10000:12000], lw=2, label="800 ms", xticks=[], ylabel="mV", ann=(50, 10, "A"))
    p2 = plot(ℬl.U0[51][4,10000:12000], lw=2, label="550 ms", xticks=[], ylabel="mV", ann=(50, 10, "B"))
    p3 = plot(ℬl.U0[1][4,10000:12000], lw=2, label="300 ms", ylabel="mV", xlabel="time (ms)", ann=(50, 10, "C"))
    p = plot(p1, p2, p3, layout=(3,1))
    return p
end

function plot_signal_perturbed()
    p1 = plot(ℬl.U0[11][4,9900:11900], lw=2, label="control", xticks=[], ylabel="mV")
    plot!(p1, ℬl.U2[11][4,9900:11900], lw=2, label="perturbed", xticks=[], ylabel="mV", ann=(50, 10, "A"), title="long qt")

    p2 = plot(ℬl.U0[11][4,9900:11900], lw=2, label=nothing, xticks=[], ylabel="mV")
    plot!(p2, ℬs.U2[11][4,9900:11900], lw=2, label=nothing, xticks=[], ylabel="mV", ann=(50, 10, "B"), title="short qt")

    p3 = plot(ℬl.U0[11][4,9900:11900], lw=2, label=nothing, ylabel="mV", xlabel="time (ms)")
    plot!(p3, ℬt.U2[11][4,9900:11900], lw=2, label=nothing, xticks=[], ylabel="mV", ann=(50, 10, "C"), title="ito")

    p = plot(p1, p2, p3, layout=(3,1))
    return p
end

function plot_neural_ode_perturbed()
    names = ["longqt", "shortqt", "ito"]
    ηη = [0.0015, 0.0018, 0.0020]
    ℬℬ = [ℬl, ℬs, ℬt]
    ps = []

    for (name, η, ℬ) in zip(names, ηη, ℬℬ)
        @info "processing $name"
        ℛ = load_model(submodel_path(models_dir, name, η))
        yq = integrate_neural_ode(ℛ.mq, ℳ, normalizer, 350)
        p = plot(ℬ.U0[11][nv,9900:11900], lw=2, label="base", xticks=[], ylabel="mV")
        plot!(p, ℬ.U2[11][nv,9900:11900], lw=2, label="pert", xticks=[], ylabel="mV")
        plot!(p, yq[nv,9900:11900], lw=2, label="gnn", xticks=[], ylabel="mV", ann=(50, 10, "A"), title="long qt")
        push!(ps, p)
    end

    return plot(ps..., layout=(3,1))
end

function plot_gates_longqt(; what=:h)
    ℛ = load_model(submodel_path(models_dir, "longqt", 0.0010))
    plot_model(ℬl, ℛ; what=what)
end

function plot_gates_shortqt(; what=:h)
    ℛ = load_model(submodel_path(models_dir, "shortqt", 0.0011))
    plot_model(ℬs, ℛ; what=what)
end

function plot_gates_ito(; what=:h)
    ℛ = load_model(submodel_path(models_dir, "ito", 0.0010))
    plot_model(ℬt, ℛ; what=what)
end

###############################################################################

function plot_model(ℬ, ℛ; what=:h)
    plt = plot(layout=(4,3), size=(600,600))
    k = [7,10,4,1,11,2,8,6,3,5]
    for i = 1:10
        plot_compare_gates(ℬ.m1, ℛ.mq, i; plt=plt, subplot=k[i], what=what)
    end
    for i in [9,12]
        plot!(plt, legend=false,grid=false,foreground_color_subplot=:white, subplot=i)
    end
    display(plt)
end

function plot_compare_gates(m1, m2, l=-1; what=:h, plt=nothing, subplot=1)
    W1, h1, ρ1, τ1 = estimate_gating(m1)
    W2, h2, ρ2, τ2 = estimate_gating(m2)

    if plt == nothing
        plt = plot(1, label=nothing)
    end

    s = String(names[mask .== 0][l])

    V = K[nv,:]

    if what == :h
        plot!(plt, V, h1[l,:], linewidth=2, label=nothing, linestyle=:dot, subplot=subplot)
        plot!(plt, V, h2[l,:], linewidth=2, label=nothing, title=s, subplot=subplot, ylims=(0,1.0))
    elseif what == :τ
        plot!(plt, V, τ1[l,:], linewidth=2, label=nothing, linestyle=:dot, subplot=subplot)
        plot!(plt, V, τ2[l,:], linewidth=2, label=nothing, title=s, subplot=subplot)
    else
        error("what not recognized")
    end
end

function estimate_gating(m; Δt=1.0)
    m = cpu(m)
    ϕ = Chain(m[1:end-1]...)
    c = m[end].cell
    Ws, Wt, bs, bt = c.Ws, c.Wt, c.bs, c.bt

    h∞ = zeros(size(Ws,1), size(K,2))
    ρ = similar(h∞)
    τ = similar(h∞)
    W = similar(K)

    for i = 1:size(K,2)
        u = normalizer(K[:,i])

        for i = 1:500
            m(u[obs])
        end

        x = ϕ(u[obs])
        W[:,i] = u
        h∞[:,i] = σ.(Ws * x .+ bs)
        ρ[:,i] = σ.(Wt * x .+ bt)
        τ[:,i] = -Δt ./ log.(ρ[:,i])
    end

    return W, h∞, ρ, τ
end
