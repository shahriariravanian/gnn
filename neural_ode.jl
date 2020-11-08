

function integrate_neural_ode(ϕ, ℳ, normalizer, cl; tspan=(0, 20000.0))
    q = copy(ℳ.p)
    q[ℳ.icl] = cl
    g! = get_f(ϕ, ℳ.obs, normalizer)
    prob = ODEProblem(g!, ℳ.u0, tspan, q)
    sol = solve(prob, TRBDF2(), dtmax=0.25, saveat=1.0)
    return Array(sol)
end

###############################################################################

function get_f(m, obs, normalizer; Δt=1f0)
    m = cpu(m)
    μ = Chain(m[1:end-1]...)
    Γ = m[end].cell

    return (duₚ, uₚ, pₚ, tₚ) -> begin
    	time = tₚ

    	# state variables:
    	Ca_i = uₚ[1]
    	r = uₚ[2]
    	d = uₚ[3]
    	V = uₚ[4]
    	fCa = uₚ[5]
    	Xs = uₚ[6]
    	m = uₚ[7]
    	f = uₚ[8]
    	g = uₚ[9]
    	K_i = uₚ[10]
    	h = uₚ[11]
    	s = uₚ[12]
    	Xr2 = uₚ[13]
    	j = uₚ[14]
    	Ca_SR = uₚ[15]
    	Xr1 = uₚ[16]
    	Na_i = uₚ[17]

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
    	i_Stim = (𝐻((time - floor(time / stim_period) * stim_period) - stim_start) * 𝐻((stim_start + stim_duration) - (time - floor(time / stim_period) * stim_period))) * -stim_amplitude
    	E_Na = ((R * T) / F) * log(Na_o / Na_i)
    	E_K = ((R * T) / F) * log(K_o / K_i)
    	E_Ks = ((R * T) / F) * log((K_o + P_kna * Na_o) / (K_i + P_kna * Na_i))
    	E_Ca = ((0.5 * (R * T)) / F) * log(Ca_o / Ca_i)
    	alpha_K1 = 0.1 / (1.0 + exp(((V - E_K) - 200.0) * 0.06))
    	beta_K1 = (exp(((V - E_K) + 100.0) * 0.0002) * 3.0 + exp(((V - E_K) - 10.0) * 0.1)) / (1.0 + exp((V - E_K) * -0.5))
    	xK1_inf = alpha_K1 / (alpha_K1 + beta_K1)
    	i_K1 = g_K1 * (xK1_inf * (sqrt(K_o / 5.4) * (V - E_K)))
    	i_Kr = g_Kr * (sqrt(K_o / 5.4) * (Xr1 * (Xr2 * (V - E_K))))
        i_Ks = g_Ks * (Xs ^ 2.0 * (V - E_Ks))
    	i_Na = g_Na * (m ^ 3.0 * (h * (j * (V - E_Na))))

    	i_b_Na = g_bna * (V - E_Na)
    	i_CaL = (((g_CaL * (d * (f * (fCa * (4.0 * (V * F ^ 2.0)))))) / (R * T)) * (Ca_i * exp((2.0 * (V * F)) / (R * T)) - Ca_o * 0.341)) / (exp((2.0 * (V * F)) / (R * T)) - 1.0)

    	alpha_fCa = 1.0 / (1.0 + (Ca_i / 0.000325) ^ 8.0)
    	beta_fCa = 0.1 / (1.0 + exp((Ca_i - 0.0005) / 0.0001))
    	gama_fCa = 0.2 / (1.0 + exp((Ca_i - 0.00075) / 0.0008))
    	fCa_inf = (alpha_fCa + (beta_fCa + (gama_fCa + 0.23))) / 1.46
    	tau_fCa = 2.0
    	d_fCa = (fCa_inf - fCa) / tau_fCa
    	i_b_Ca = g_bca * (V - E_Ca)

    	i_to = g_to * (r * (s * (V - E_K)))

    	i_NaK = ((((P_NaK * K_o) / (K_o + K_mk)) * Na_i) / (Na_i + K_mNa)) / (1.0 + (exp((-0.1 * (V * F)) / (R * T)) * 0.1245 + exp((-V * F) / (R * T)) * 0.0353))
    	i_NaCa = (K_NaCa * (exp((gamma * (V * F)) / (R * T)) * (Na_i ^ 3.0 * Ca_o) - exp(((gamma - 1.0) * (V * F)) / (R * T)) * (Na_o ^ 3.0 * (Ca_i * alpha)))) / ((Km_Nai ^ 3.0 + Na_o ^ 3.0) * ((Km_Ca + Ca_o) * (1.0 + K_sat * exp(((gamma - 1.0) * (V * F)) / (R * T)))))
    	i_p_Ca = (g_pCa * Ca_i) / (Ca_i + K_pCa)
    	i_p_K = (g_pK * (V - E_K)) / (1.0 + exp((25.0 - V) / 5.98))
    	i_rel = ((a_rel * Ca_SR ^ 2.0) / (b_rel ^ 2.0 + Ca_SR ^ 2.0) + c_rel) * (d * g)
    	i_up = Vmax_up / (1.0 + K_up ^ 2.0 / Ca_i ^ 2.0)
    	i_leak = V_leak * (Ca_SR - Ca_i)
    	g_inf = 𝐻((0.00035 - Ca_i) - 2.220446049250313e-16) * (1.0 / (1.0 + (Ca_i / 0.00035) ^ 6.0)) + (1 - 𝐻((0.00035 - Ca_i) - 2.220446049250313e-16)) * (1.0 / (1.0 + (Ca_i / 0.00035) ^ 16.0))
    	d_g = (g_inf - g) / tau_g
    	Ca_i_bufc = 1.0 / (1.0 + (Buf_c * K_buf_c) / (Ca_i + K_buf_c) ^ 2.0)
    	Ca_sr_bufsr = 1.0 / (1.0 + (Buf_sr * K_buf_sr) / (Ca_SR + K_buf_sr) ^ 2.0)

        #####################################################################

        u = normalizer(uₚ)
        x = μ(u[obs])
        h∞ = σ.(Γ.Ws * x .+ Γ.bs)
        ρ = σ.(Γ.Wt * x .+ Γ.bt)
        τ = -Δt ./ log.(ρ)

        xr1_inf, tau_xr1 = h∞[10], τ[10]
        xr2_inf, tau_xr2 = h∞[8], τ[8]
        xs_inf, tau_xs = h∞[3], τ[3]
        m_inf, tau_m = h∞[4], τ[4]
        h_inf, tau_h = h∞[6], τ[6]
        j_inf, tau_j = h∞[9], τ[9]
        d_inf, tau_d = h∞[2], τ[2]
        f_inf, tau_f = h∞[5], τ[5]
        s_inf, tau_s = h∞[7], τ[7]
        r_inf, tau_r = h∞[1], τ[1]
        # fCa_inf, tau_fCa = h∞[5], τ[5]

        d_fCa = (fCa_inf - fCa) / tau_fCa

        #####################################################################


    	# system of ODEs:
    	∂V = -((i_K1 + (i_to + (i_Kr + (i_Ks + (i_CaL + (i_NaK + (i_Na + (i_b_Na + (i_NaCa + (i_b_Ca + (i_p_K + (i_p_Ca + i_Stim)))))))))))))
    	∂Xr1 = (xr1_inf - Xr1) / tau_xr1
    	∂Xr2 = (xr2_inf - Xr2) / tau_xr2
    	∂Xs = (xs_inf - Xs) / tau_xs
    	∂m = (m_inf - m) / tau_m
    	∂h = (h_inf - h) / tau_h
    	∂j = (j_inf - j) / tau_j
    	∂d = (d_inf - d) / tau_d
    	∂f = (f_inf - f) / tau_f
    	∂fCa = (1 - 𝐻((fCa_inf - fCa) - 2.220446049250313e-16) * 𝐻((V - -60.0) - 2.220446049250313e-16)) * d_fCa
    	∂s = (s_inf - s) / tau_s
    	∂r = (r_inf - r) / tau_r
    	∂g = (1 - 𝐻((g_inf - g) - 2.220446049250313e-16) * 𝐻((V - -60.0) - 2.220446049250313e-16)) * d_g
    	∂Ca_i = Ca_i_bufc * (((i_leak - i_up) + i_rel) - (((i_CaL + (i_b_Ca + i_p_Ca)) - i_NaCa * 2.0) / (2.0 * (V_c * F))) * Cm)
    	∂Ca_SR = ((Ca_sr_bufsr * V_c) / V_sr) * (i_up - (i_rel + i_leak))
    	∂Na_i = -((i_Na + (i_b_Na + (i_NaK * 3.0 + i_NaCa * 3.0))) * Cm) / (V_c * F)
    	∂K_i = -(((i_K1 + (i_to + (i_Kr + (i_Ks + (i_p_K + i_Stim))))) - i_NaK * 2.0) * Cm) / (V_c * F)

    	# state variables:
    	duₚ[1] = ∂Ca_i
    	duₚ[2] = ∂r
    	duₚ[3] = ∂d
    	duₚ[4] = ∂V
    	duₚ[5] = ∂fCa
    	duₚ[6] = ∂Xs
    	duₚ[7] = ∂m
    	duₚ[8] = ∂f
    	duₚ[9] = ∂g
    	duₚ[10] = ∂K_i
    	duₚ[11] = ∂h
    	duₚ[12] = ∂s
    	duₚ[13] = ∂Xr2
    	duₚ[14] = ∂j
    	duₚ[15] = ∂Ca_SR
    	duₚ[16] = ∂Xr1
    	duₚ[17] = ∂Na_i
    	nothing
    end

end
