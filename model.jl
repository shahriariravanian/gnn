using BSON
using Plots
using Printf: @sprintf

include("tentusscher.jl")

const mask = [1,0,0,1,1,0,0,0,1,1,0,0,0,0,1,0,1]
const names = [:Ca_i, :r, :d, :V, :fCa, :Xs, :m, :f, :g,
               :K_i, :h, :s, :Xr2, :j, :Ca_SR, :Xr1, :Na_i]


struct IonicModel{F,V}
    f!::F
    p::V
    u0::V
    mask::Array{Int}
    u0_names::Array{Symbol}
    p_names::Array{Symbol}
    obs::Array{Int}
    icl::Int
    nv::Int
end

function TenTusscherModel()
    return IonicModel(
        f!,
        p,
        u0,
        [1,0,0,1,1,0,0,0,1,1,0,0,0,0,1,0,1],
        [:Ca_i, :r, :d, :V, :fCa, :Xs, :m, :f, :g, :K_i, :h, :s, :Xr2, :j,
         :Ca_SR, :Xr1, :Na_i],
        [:stim_start, :g_pK, :g_bna, :K_mNa, :b_rel, :g_Ks, :K_pCa, :g_Kr,
         :Na_o, :K_up, :g_pCa, :alpha, :stim_amplitude, :V_leak, :Buf_c,
         :g_CaL, :F, :T, :P_kna, :g_bca, :Km_Ca, :c_rel, :K_buf_sr, :Km_Nai,
         :K_sat, :a_rel, :tau_g, :Cm, :g_to, :P_NaK, :g_K1, :stim_duration,
         :K_mk, :Ca_o, :stim_period, :V_sr, :V_c, :K_o, :K_buf_c, :Buf_sr,
         :g_Na, :Vmax_up, :K_NaCa, :R, :gamma],
        [1,4],
        35,
        4
    )
end

function modify_model(ℳ, params)
    q = copy(ℳ.p)
    for param in params
        i = findfirst(x -> x==first(param), ℳ.p_names)
        q[i] *= last(param)
    end

    return IonicModel(
        ℳ.f!,
        q,
        ℳ.u0,
        ℳ.mask,
        ℳ.u0_names,
        ℳ.p_names,
        ℳ.obs,
        ℳ.icl,
        ℳ.nv
    )
end

model_path(dir, prefix) = joinpath(dir, prefix * ".bson")

function submodel_path(dir, prefix, η)
    name = @sprintf "%s_η%.4f.bson" prefix η
    return joinpath(dir, name)
end

function save_model(ℬ, path)
    X = cpu(ℬ)
    BSON.@save path X
end

function load_model(path)
    X = BSON.load(path)
    return X[first(keys(X))]
end
