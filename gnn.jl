#=
    The main Gating Neural Network (GNN) definition
=#

using Flux

mutable struct GNNCell{A,V}
    Δt::Float32
    Ws::A
    Wt::A
    bs::V
    bt::V
    h::V
end

function GNNCell(in::Integer, gate::Integer; init = Flux.glorot_uniform, Δt=1.0)
    return GNNCell( Float32(Δt),
                    init(gate, in), init(gate, in),
                    init(gate), init(gate),
                    init(gate))
end

function (m::GNNCell)(h, x)
    Ws, Wt, bs, bt, Δt = m.Ws, m.Wt, m.bs, m.bt, m.Δt

    h∞ = σ.(Ws * x .+ bs)           # the steady-state values of the gates
    ρ = σ.(Wt * x .+ bt)            # the
    h = h .* ρ .+ h∞ .* (1 .- ρ)
    return h, h
end

Flux.hidden(m::GNNCell) = m.h

Flux.trainable(m::GNNCell) = (m.Ws, m.Wt, m.bs, m.bt)

Flux.@functor GNNCell

function Base.show(io::IO, m::GNNCell)
    in, gate = size(m.Ws)
    print(io, "GNNCell(", gate, ",", in, ")")
end

GNN(a...; ka...) = Flux.Recur(GNNCell(a...; ka...))
