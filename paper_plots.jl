using Plots
using Flux

include("model.jl")
include("gnn.jl")
include("prepare.jl")
include("neural_ode.jl")

const models_dir="models"

const ℳ = TenTusscherModel()
const obs = ℳ.obs
const nv = ℳ.nv

if ! @isdefined ℬl
    @info "loading longqt"
    const ℬl = load_model(model_path(models_dir, "longqt"))
    @info "calculating normalizer"
    const K = voltage_clamp(Float32, ℳ)
    const normalizer, denormalizer = create_normalizer(ℬl.U0, K)
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
    ps = []
    nc = 1

    for (cl, a) in zip([800, 550, 320], ["A", "B", "C"])
        k = 1 + (cl - 300) ÷ 5
        p = plot(ℬl.U0[k][nv,10000:12000], lw=3, label="$cl ms", xticks=[], ylabel="Vm (mV)", ann=(50, 10, a))
        q = twinx(p)
        plot!(q, ℬl.U0[k][nc,10000:12000], lw=3, label=nothing, xticks=[], yticks=[], ylabel="[Ca] (a.u.)", color=:orange)
        push!(ps, p)
    end

    plot!(ps[end], xticks=[0,500,1000,1500,2000], xlabel="time (ms)")

    p = plot(ps..., layout=(3,1), size=(1200,800))
    return p
end

function plot_signal_perturbed(; k=11)
    p1 = plot(ℬl.U0[k][4,9900:11900], lw=3, label="control", xticks=[], ylabel="Vm (mV)")
    plot!(p1, ℬl.U2[k][4,9900:11900], lw=3, label="perturbed", xticks=[], ann=(50, 10, "A"), title="long qt")

    p2 = plot(ℬl.U0[k][4,9900:11900], lw=3, label=nothing, xticks=[], ylabel="Vm (mV)")
    plot!(p2, ℬs.U2[k][4,9900:11900], lw=3, label=nothing, xticks=[], ann=(50, 10, "B"), title="short qt")

    p3 = plot(ℬl.U0[k][4,9900:11900], lw=3, label=nothing, ylabel="Vm (mV)", xlabel="time (ms)", )
    plot!(p3, ℬt.U2[k][4,9900:11900], lw=3, label=nothing, xticks=[], ann=(50, 10, "C"), title="ito")

    plot!(p3, xticks=[0,500,1000,1500,2000], xlabel="time (ms)")

    p = plot(p1, p2, p3, layout=(3,1), size=(1200,800))
    return p
end

function plot_neural_ode(; cl=350)
    names = ["longqt", "shortqt", "ito"]
    ηη = [0.0016, 0.0018, 0.0015]
    ℬℬ = [ℬl, ℬs, ℬt]
    ps = []
    k = 1 + (cl - 300) ÷ 5

    for (name, η, ℬ) in zip(names, ηη, ℬℬ)
        @info "processing $name"
        ℛ = load_model(submodel_path(models_dir, name, η))
        yq = integrate_neural_ode(ℛ.mq, ℳ, normalizer, cl)

        p = plot(ℬ.U0[k][nv,9900:11900], lw=3, label="ODE (base)", xticks=[], ylabel="mV")
        plot!(p, ℬ.U2[k][nv,9900:11900], lw=3, label="ODE (perturbed)", xticks=[], ylabel="mV")
        plot!(p, yq[nv,9900:11900], lw=3, label="GNN", xticks=[], ylabel="mV", ann=(50, 10, "A"), title=name)

        push!(ps, p)
    end

    return plot(ps..., layout=(3,1), size=(1200,800))
end

function plot_neural_ode_extra(; cl=350)
    names = ["longqt", "shortqt", "ito"]
    ηη = [0.0016, 0.0018, 0.0015]
    ℬℬ = [ℬl, ℬs, ℬt]
    ps = []
    k = 1 + (cl - 300) ÷ 5

    for (name, η, ℬ) in zip(names, ηη, ℬℬ)
        @info "processing $name"
        ℛ = load_model(submodel_path(models_dir, name, η))
        yq = integrate_neural_ode(ℛ.mq, ℳ, normalizer, cl)

        # p = plot(ℬ.U0[k][nv,9900:11900], lw=2, label="base", xticks=[], ylabel="mV")
        # plot!(p, ℬ.U2[k][nv,9900:11900], lw=2, label="pert", xticks=[], ylabel="mV")
        p = plot(yq[nv,9900:11900], lw=2, label="gnn", xticks=[], ylabel="mV", ann=(50, 10, "A"), title=name)
        push!(ps, p)
    end

    return plot(ps..., layout=(3,1))
end

function plot_gates_longqt(; what=:h, η=0.0010)
    ℛ = load_model(submodel_path(models_dir, "longqt", η))
    plot_model(ℬl, ℛ; what=what)
end

function plot_gates_shortqt(; what=:h, η=0.0011)
    ℛ = load_model(submodel_path(models_dir, "shortqt", η))
    plot_model(ℬs, ℛ; what=what)
end

function plot_gates_ito(; what=:h, η=0.0010)
    ℛ = load_model(submodel_path(models_dir, "ito", η))
    plot_model(ℬt, ℛ; what=what)
end

function plot_currents_longqt(; what=:h, η=0.0010)
    ℛ = load_model(submodel_path(models_dir, "longqt", η))
    plot_currents(ℬl, ℛ)
end

function plot_currents_shortqt(; what=:h, η=0.0016)
    ℛ = load_model(submodel_path(models_dir, "shortqt", η))
    plot_currents(ℬs, ℛ)
end

function plot_currents_ito(; what=:h, η=0.0010)
    ℛ = load_model(submodel_path(models_dir, "ito", η))
    plot_currents(ℬt, ℛ)
end

###############################################################################

function plot_model(ℬ, ℛ; what=:h)
    plt = plot(layout=(4,3), size=(600,600))
    k = [7,10,4,1,11,2,8,6,3,5]
    for i = 1:10
        plot_compare_gates(ℛ.mp, ℛ.mq, i; plt=plt, subplot=k[i], what=what)
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
        # plot!(plt, V, h2[l,:] .- h1[l,:], linewidth=2, label=nothing, title=s, subplot=subplot, ylims=(-1.0,1.0))
    elseif what == :τ
        plot!(plt, V, τ1[l,:], linewidth=2, label=nothing, linestyle=:dot, subplot=subplot)
        plot!(plt, V, τ2[l,:], linewidth=2, label=nothing, title=s, subplot=subplot)
    else
        error("what not recognized")
    end
end

function plot_combined_model(ℬ, ℛ; what=:h)
    W1, h∞1, ρ1, τ1 = estimate_gating(ℬ.m1)
    W2, h∞2, ρ2, τ2 = estimate_gating(ℛ.mq)

    m₁, m₂ = h∞1[4,:], h∞2[4,:]
    h₁, h₂ = h∞1[6,:], h∞2[6,:]
    j₁, j₂ = h∞1[4,:], h∞2[9,:]
    Na₁, Na₂ = m₁.^3 .* h₁ .* j₁, m₂.^3 .* h₂ .* j₂

    d₁, d₂ = h∞1[2,:], h∞2[2,:]
    f₁, f₂ = h∞1[5,:], h∞2[5,:]
    Ca₁, Ca₂ = d₁ .* f₁, d₂ .* f₂

    xs₁, xs₂ = h∞1[3,:], h∞2[3,:]
    XS₁, XS₂ = xs₁.^2, xs₂.^2

    xr1₁, xr1₂ = h∞1[10,:], h∞2[10,:]
    xr2₁, xr2₂ = h∞1[8,:], h∞2[8,:]
    XR₁, XR₂ = xr1₁ .* xr2₁, xr1₂ .* xr2₂

    r₁, r₂ = h∞1[1,:], h∞2[1,:]
    s₁, s₂ = h∞1[7,:], h∞2[7,:]
    Kto₁, Kto₂ = r₁ .* s₁, r₂ .* s₂

    V = K[nv,:]

    plt = plot(layout=(2,2), size=(600,600))

    plot!(plt, V, Na₁, linewidth=2, label=nothing, linestyle=:dot, subplot=1)
    plot!(plt, V, Na₂, linewidth=2, label=nothing, subplot=1)

    display(plt)
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

    u = normalizer(K)

    for i = 1:500
        m(u[obs,:])
    end

    x = ϕ(u[obs,:])
    W = u
    h∞ = σ.(Ws * x .+ bs)
    ρ = σ.(Wt * x .+ bt)
    τ = -Δt ./ log.(ρ)

    # for i = 1:size(K,2)
    #     u = normalizer(K[:,i])
    #
    #     for j = 1:500
    #         m(u[obs])
    #     end
    #
    #     x = ϕ(u[obs])
    #     W[:,i] = u
    #     h∞[:,i] = σ.(Ws * x .+ bs)
    #     ρ[:,i] = σ.(Wt * x .+ bt)
    #     τ[:,i] = -Δt ./ log.(ρ[:,i])
    # end

    return W, h∞, ρ, τ
end

function plot_currents(ℬ, ℛ; k=30, rng=7001:9000)
    X₀ = estimate_currents(ℬ.U0, ℛ.mp, p; k=k)
    X₂ = estimate_currents(ℬ.U2, ℛ.mq, p; k=k)

    plt = plot(layout=(6,1), size=(1200,800))
    titles = ["Vm", "I_Na", "I_Ca", "I_Kr", "I_Ks", "I_to"]

    plot!(plt, X₀[1][rng], subplot=1, lw=3, ylabel=titles[1], label="ODE", color=:blue)
    plot!(plt, X₂[1][rng], subplot=1, lw=3, label="GNN", xticks=[], yticks=[], color=:red)

    for i = 2:6
        plot!(plt, X₀[i][rng], subplot=i, lw=3, ylabel=titles[i], label=nothing, color=:blue)
        plot!(plt, X₂[i][rng], subplot=i, lw=3, label=nothing, xticks=[], yticks=[], color=:red)
    end

    plot!(plt, subplot=6, xlabel="time (ms)", xticks=[0,500,1000,1500,2000])

    display(plt)
end

############################################################################

function estimate_currents(U, m, pₚ; k=10)
    m = cpu(m)
    n = size(U[1],2)
    xₚ = hcat([U[k][:,i] for i=1:n]...)
    hₚ = hcat([m(normalizer(xₚ[:,i])[obs]) for i=1:n]...)

    Vm = zeros(n)
    I_Na = zeros(n)
    I_CaL = zeros(n)
    I_Kr = zeros(n)
    I_Ks = zeros(n)
    I_to = zeros(n)

    for i = 1:n
        Ca_i = xₚ[1,i]
        r = hₚ[1,i]
        d = hₚ[2,i]
        V = xₚ[4,i]
        fCa = xₚ[5,i]
        Xs = hₚ[3,i]
        m = hₚ[4,i]
        f = hₚ[5,i]
        g = xₚ[9,i]
        K_i = xₚ[10,i]
        h = hₚ[6,i]
        s = hₚ[7,i]
        Xr2 = hₚ[8,i]
        j = hₚ[9,i]
        Ca_SR = xₚ[15,i]
        Xr1 = hₚ[10,i]
        Na_i = xₚ[17,i]

        # parameters:
        stim_start = pₚ[1]
        g_pK = pₚ[2]
        g_bna = pₚ[3]
        K_mNa = pₚ[4]
        b_rel = pₚ[5]
        g_Ks = pₚ[6]
        K_pCa = pₚ[7]
        g_Kr = pₚ[8]
        Na_o = pₚ[9]
        K_up = pₚ[10]
        g_pCa = pₚ[11]
        alpha = pₚ[12]
        stim_amplitude = pₚ[13]
        V_leak = pₚ[14]
        Buf_c = pₚ[15]
        g_CaL = pₚ[16]
        F = pₚ[17]
        T = pₚ[18]
        P_kna = pₚ[19]
        g_bca = pₚ[20]
        Km_Ca = pₚ[21]
        c_rel = pₚ[22]
        K_buf_sr = pₚ[23]
        Km_Nai = pₚ[24]
        K_sat = pₚ[25]
        a_rel = pₚ[26]
        tau_g = pₚ[27]
        Cm = pₚ[28]
        g_to = pₚ[29]
        P_NaK = pₚ[30]
        g_K1 = pₚ[31]
        stim_duration = pₚ[32]
        K_mk = pₚ[33]
        Ca_o = pₚ[34]
        stim_period = pₚ[35]
        V_sr = pₚ[36]
        V_c = pₚ[37]
        K_o = pₚ[38]
        K_buf_c = pₚ[39]
        Buf_sr = pₚ[40]
        g_Na = pₚ[41]
        Vmax_up = pₚ[42]
        K_NaCa = pₚ[43]
        R = pₚ[44]
        gamma = pₚ[45]

        # algebraic equations:
        E_Na = ((R * T) / F) * log(Na_o / Na_i)
        E_K = ((R * T) / F) * log(K_o / K_i)
        E_Ks = ((R * T) / F) * log((K_o + P_kna * Na_o) / (K_i + P_kna * Na_i))
        E_Ca = ((0.5 * (R * T)) / F) * log(Ca_o / Ca_i)

        I_Kr[i] = g_Kr * (sqrt(K_o / 5.4) * (Xr1 * (Xr2 * (V - E_K))))
        I_Ks[i] = g_Ks * (Xs ^ 2.0 * (V - E_Ks))
        I_Na[i] = g_Na * (m ^ 3.0 * (h * (j * (V - E_Na))))
        I_CaL[i] = (((g_CaL * (d * (f * (fCa * (4.0 * (V * F ^ 2.0)))))) / (R * T)) * (Ca_i * exp((2.0 * (V * F)) / (R * T)) - Ca_o * 0.341)) / (exp((2.0 * (V * F)) / (R * T)) - 1.0)
        I_to[i] = g_to * (r * (s * (V - E_K)))
        Vm[i] = V
    end

    # Vm = x[4,:]
    # I_Na = h[4,:].^3 .* h[6,:] .* h[9,:]
    # I_Ca = h[2,:] .* h[5,:]
    # I_Ks = h[3,:].^2
    # I_Kr = h[10,:] .* h[8,:]
    # I_to = h[1,:] .* h[7,:]

    return Vm, I_Na, I_CaL, I_Kr, I_Ks, I_to
end
