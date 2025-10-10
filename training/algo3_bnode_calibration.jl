"""
Algorithm 3: BNODE Posterior Calibration (from Structure-Preserving Models with Calibrated Uncertainty)

Steps:
1. Input: Posterior samples {θ⁽ˢ⁾, σ⁽ˢ⁾}⁽ˢ⁾ₛ₌₁ (S=100), validation and test data
2. For each scenario i, predict with all S draws to get μᵢ (mean), sᵢ (stddev)
3. Search α in [0.5, 2.0] (80 steps) to minimize (cov₅₀ − 0.5)² + (cov₉₀ − 0.9)²
4. Use bands μ ± z·s, with z₀.₇₅ ≈ 0.674, z₀.₉₅ ≈ 1.645
5. Apply to test: compute all test metrics using optimal α*
"""
#!/usr/bin/env julia

using BSON, JSON3, MCMCChains
using Accessors
using StaticArrays, Statistics, LinearAlgebra, Printf
using Distributions
using CSV, DataFrames
using AbstractPPL
using OrdinaryDiffEq
using Random

# ------------------------------- NN layout (paper-aligned) ---------------------------------

struct DenseSimple{T}
    W::Matrix{T}   # 5×4
    b::Vector{T}   # 5
end

σact(x) = tanh.(x)

@inline function forward(d::DenseSimple, z::SVector{4,Float64})
    return sum(σact.(d.W * z .+ d.b))
end

function unpack_two_nets(θ::Vector{Float64})
    @assert length(θ) == 50 "θ length $(length(θ)) != 50 (expected)"
    # Net 1
    W1 = Matrix(reshape(@view(θ[1:20]), 5, 4))
    b1 = copy(@view(θ[21:25]))
    # Net 2
    W2 = Matrix(reshape(@view(θ[26:45]), 5, 4))
    b2 = copy(@view(θ[46:50]))
    return DenseSimple(W1,b1), DenseSimple(W2,b2)
end

# ------------------------------- Data loaders ---------------------------------

Base.@kwdef mutable struct Exog
    t::Vector{Float64}
    u::Vector{Float64}
    d::Vector{Float64}
    Pgen::Vector{Float64}
    Pload::Vector{Float64}
end

@inline function lininterp(x::Vector{Float64}, y::Vector{Float64}, τ::Float64)
    if τ <= x[1]
        return y[1]
    elseif τ >= x[end]
        return y[end]
    end
    i = searchsortedlast(x, τ)
    i = clamp(i, 1, length(x) - 1)
    x0, x1 = x[i], x[i + 1]
    y0, y1 = y[i], y[i + 1]
    w = (τ - x0) / max(x1 - x0, eps())
    return (1 - w) * y0 + w * y1
end

function build_scenarios(df::DataFrame; split::Symbol)
    @assert split in (:val, :test)
    groups = groupby(df, :scenario)
    scenarios = []
    for g in groups
        # time vector fallback if :t missing
        t = if :t in names(g)
            collect(Float64, g.t)
        else
            n = nrow(g)
            collect(range(0.0, step=0.05, length=n))
        end
        u = collect(Float64, g.u)
        d = collect(Float64, g.d)
        Pgen = collect(Float64, g.Pgen)
        Pload = collect(Float64, g.Pload)
        x1v = collect(Float64, g.x1)
        x2v = collect(Float64, g.x2)
        @assert length(t) == length(u) == length(d) == length(Pgen) == length(Pload) == length(x1v) == length(x2v) > 1
        ex = Exog(t = t, u = u, d = d, Pgen = Pgen, Pload = Pload)
        u0 = @SVector [x1v[1], x2v[1]]
        y_true = reduce(hcat, (SVector{2,Float64}(x1v[i], x2v[i]) for i in eachindex(t)))
        tspan = (t[1], t[end])
        push!(scenarios, (ex, u0, tspan, t, y_true))
    end
    return scenarios
end

function load_validation_scenarios()
    @assert isfile("data/validation_roadmap.csv") "Missing data/validation_roadmap.csv"
    df = CSV.read("data/validation_roadmap.csv", DataFrame)
    return build_scenarios(df; split=:val)
end

function load_test_scenarios()
    path = isfile("data/test_roadmap.csv") ? "data/test_roadmap.csv" : "data/validation_roadmap.csv"
    df = CSV.read(path, DataFrame)
    return build_scenarios(df; split=:test)
end

# ------------------------------- Integrator (Tsit5) ---------------------------------

function rhs!(du::Vector{Float64}, u::Vector{Float64}, p, τ::Float64)
    n1, n2, ex = p
    x1, x2 = u
    u_in   = lininterp(ex.t, ex.u,    τ)
    d_in   = lininterp(ex.t, ex.d,    τ)
    Pgen   = lininterp(ex.t, ex.Pgen, τ)
    Pload  = lininterp(ex.t, ex.Pload,τ)
    z1 = @SVector [x1, x2, u_in,  d_in]
    z2 = @SVector [x1, x2, Pgen,  Pload]
    du[1] = forward(n1, z1)
    du[2] = forward(n2, z2)
    return nothing
end

function simulate_draw(θ_vec::Vector{Float64}, ex::Exog, u0::SVector{2,Float64},
                       tspan::Tuple{Float64,Float64}, saveat::Vector{Float64})
    n1, n2 = unpack_two_nets(θ_vec)
    prob = ODEProblem(rhs!, collect(u0), tspan, (n1, n2, ex))
    sol = solve(prob, Tsit5(); saveat=saveat, reltol=1e-6, abstol=1e-6)
    Y = reduce(hcat, sol.u)  # 2×T
    @assert all(isfinite, Y) "Non-finite values in trajectory"
    return Y
end

# ------------------------------- Metrics ---------------------------------

function central_interval(vals::Vector{Float64}, q::Float64)
    K = length(vals)
    sort!(vals)
    lo_p = (1 - q) / 2
    hi_p = 1 - lo_p
    lo_i = clamp(lo_p * (K - 1) + 1, 1.0, K*1.0)
    hi_i = clamp(hi_p * (K - 1) + 1, 1.0, K*1.0)
    function interp_at(idx)
        i = floor(Int, idx)
        δ = idx - i
        if i >= K
            return vals[K]
        end
        return vals[i] + δ * (vals[i+1] - vals[i])
    end
    return interp_at(lo_i), interp_at(hi_i)
end

function coverage_from_draws(y_true::AbstractMatrix, Ydraws::Vector{<:AbstractMatrix}; q::Float64)
    K = length(Ydraws); T = size(y_true, 2)
    covered = 0; total = 2*T
    for j in 1:2, t in 1:T
        vals = [Ydraws[k][j, t] for k in 1:K]
        lo, hi = central_interval(vals, q)
        covered += (lo <= y_true[j, t] <= hi) ? 1 : 0
    end
    return covered / total
end

function summarize_draws(Ydraws::Vector{Matrix{Float64}})
    K = length(Ydraws); T = size(Ydraws[1], 2)
    μ = zeros(2, T); s = zeros(2, T)
    for j in 1:2, t in 1:T
        vals = [Ydraws[k][j, t] for k in 1:K]
        μ[j, t] = mean(vals); s[j, t] = std(vals)
    end
    return μ, s
end

function gaussian_coverage(y_true::AbstractMatrix, μ::AbstractMatrix, s::AbstractMatrix, q::Float64; α::Float64=1.0)
    @assert size(y_true) == size(μ) == size(s)
    T = size(y_true, 2)
    covered = 0; total = 2*T
    z = quantile(Normal(), 0.5 + q/2)
    for j in 1:2, t in 1:T
        st = max(α * s[j, t], 1e-12)
        lo = μ[j, t] - z * st
        hi = μ[j, t] + z * st
        covered += (lo <= y_true[j, t] <= hi) ? 1 : 0
    end
    return covered / total
end

function nll_gaussian(y_true::AbstractMatrix, μ::AbstractMatrix, s::AbstractMatrix; α::Float64=1.0)
    @assert size(y_true) == size(μ) == size(s)
    nll = 0.0
    for j in 1:size(y_true,1), t in 1:size(y_true,2)
        st = max(α * s[j, t], 1e-12)
        nll += log(st) + 0.5 * ((y_true[j, t] - μ[j, t]) / st)^2
    end
    return nll
end

# ------------------------------- Posterior handling ---------------------------------

function load_posterior_chain()
    bson_candidates = [
        "checkpoints/bnode_posterior.bson",
        "results/bnode_chain_with_diagnostics.bson",
        "checkpoints/bnode_chain_with_diagnostics.bson"
    ]
    json_candidates = [
        "checkpoints/bnode_posterior.json"
    ]
    idx = findfirst(isfile, bson_candidates)
    path = idx === nothing ? nothing : bson_candidates[idx]
    if path === nothing
        idx2 = findfirst(isfile, json_candidates)
        idx2 !== nothing || error("No posterior file found. Tried BSON: $(join(bson_candidates, ", ")) and JSON: $(join(json_candidates, ", "))")
        path = json_candidates[idx2]
    end
    if endswith(path, ".bson")
        D = BSON.load(path)
        @assert haskey(D, :chain) "BSON must contain :chain (MCMCChains.Chains)"
        ch = D[:chain]
        namesyms = names(ch)
        pnames = string.(namesyms)
        colmap = Dict(pnames[i] => i for i in eachindex(pnames))
        θ_names = ["θ[$i]" for i in 1:50]
        @assert all(name -> haskey(colmap, name), θ_names) "Posterior missing θ[1:50] columns"
        θ_cols = map(n -> colmap[n], θ_names)
        @assert haskey(colmap, "σ") "Posterior missing σ column"
        σ_col = colmap["σ"]
        A = Array(ch)
        if ndims(A) == 3
            iters, vars, nch = size(A)
            A = reshape(A, iters * nch, vars)
        end
        return A, θ_cols, σ_col
    else
        D = JSON3.read(read(path, String))
        @assert haskey(D, :data) && haskey(D, :keys) "Malformed JSON posterior"
        data = D[:data]
        @assert haskey(data, :chain) "JSON posterior missing :chain payload"
        chmeta = data[:chain]
        @assert haskey(chmeta, :names) && haskey(chmeta, :size) "JSON chain missing names/size"
        pnames = Vector{String}(chmeta[:names])
        colmap = Dict(pnames[i] => i for i in eachindex(pnames))
        θ_names = ["θ[$i]" for i in 1:50]
        @assert all(name -> haskey(colmap, name), θ_names) "Posterior missing θ[1:50] columns"
        θ_cols = map(n -> colmap[n], θ_names)
        @assert haskey(colmap, "σ") "Posterior missing σ column"
        σ_col = colmap["σ"]
        error("JSON posterior contains only metadata (no draws array). Provide a BSON with full chain to compute calibration.")
    end
end

# ------------------------------- Main pipeline ---------------------------------

function simulate_set(scenarios, A, θ_cols; ndraws::Int, seed::Int=42)
    Random.seed!(seed)
    iters = size(A, 1)
    draws_idx = rand(1:iters, min(ndraws, iters))
    cov50s = Float64[]; cov90s = Float64[]; nlls = Float64[]
    μs = Vector{Matrix{Float64}}(); ss = Vector{Matrix{Float64}}()
    for (ex, u0, tspan, saveat, ytrue) in scenarios
        Ydraws = Matrix{Float64}[]
        for i in draws_idx
            θ = vec(A[i, θ_cols])
            Y = simulate_draw(θ, ex, u0, tspan, saveat)
            push!(Ydraws, Y)
        end
        push!(cov50s, coverage_from_draws(ytrue, Ydraws; q=0.5))
        push!(cov90s, coverage_from_draws(ytrue, Ydraws; q=0.9))
        μ, s = summarize_draws(Ydraws)
        push!(μs, μ); push!(ss, s)
        push!(nlls, nll_gaussian(ytrue, μ, s; α=1.0))
    end
    return cov50s, cov90s, μs, ss, nlls
end

function fit_alpha(VAL; targets=(0.5, 0.9))
    val_cov50, val_cov90, val_μs, val_ss, val_nlls, val_y = VAL
    bestα, bestL = 1.0, Inf
    for α in range(0.5, 2.0; length=80)
        cov50 = mean([gaussian_coverage(val_y[i], val_μs[i], val_ss[i], 0.5; α=α) for i in eachindex(val_y)])
        cov90 = mean([gaussian_coverage(val_y[i], val_μs[i], val_ss[i], 0.9; α=α) for i in eachindex(val_y)])
        L = (cov50 - 0.5)^2 + (cov90 - 0.9)^2
        if L < bestL
            bestL = L; bestα = α
        end
    end
    return bestα
end

function main()
    # 1) Load posterior
    A, θ_cols, σ_col = load_posterior_chain()
    @assert length(θ_cols) == 50 "Expected exactly 50 θ columns"

    # 2) Load scenarios
    val_scenarios = load_validation_scenarios()
    tst_scenarios = load_test_scenarios()
    @assert !isempty(val_scenarios) && !isempty(tst_scenarios)

    # Extract y_true matrices for alpha fitting
    val_y = [sc[5] for sc in val_scenarios]
    tst_y = [sc[5] for sc in tst_scenarios]

    # 3) Simulate posterior draws
    ndraws = min(100, size(A, 1))
    val_cov50, val_cov90, val_μs, val_ss, val_nlls = simulate_set(val_scenarios, A, θ_cols; ndraws=ndraws)
    tst_cov50, tst_cov90, tst_μs, tst_ss, tst_nlls_pre = simulate_set(tst_scenarios, A, θ_cols; ndraws=ndraws)

    # 4) Fit alpha on validation using Gaussian summary
    α = fit_alpha((val_cov50, val_cov90, val_μs, val_ss, val_nlls, val_y))

    # 5) Compute post-calibration coverage and NLL on test
    cov50_post = mean([gaussian_coverage(tst_y[i], tst_μs[i], tst_ss[i], 0.5; α=α) for i in eachindex(tst_y)])
    cov90_post = mean([gaussian_coverage(tst_y[i], tst_μs[i], tst_ss[i], 0.9; α=α) for i in eachindex(tst_y)])
    tst_nll_post = mean([nll_gaussian(tst_y[i], tst_μs[i], tst_ss[i]; α=α) for i in eachindex(tst_y)])

    out = Dict(
        "coverage_50_pre" => mean(tst_cov50),
        "coverage_90_pre" => mean(tst_cov90),
        "coverage_50_post" => cov50_post,
        "coverage_90_post" => cov90_post,
        "nll_pre" => mean(tst_nlls_pre),
        "nll_post" => tst_nll_post,
        "alpha" => α,
        "draws_used" => ndraws,
        "width" => 5,
        "per_net_inputs" => 4
    )

    mkpath("results")
    open("results/bnode_calibrated.json", "w") do io
        JSON3.write(io, out)
    end
    println("Wrote results/bnode_calibrated.json")
end

main()
