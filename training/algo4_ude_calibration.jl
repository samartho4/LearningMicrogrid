#!/usr/bin/env julia

using Statistics, JSON3, Dates
using Printf
using CSV, DataFrames
using Plots

rmse(yhat, y) = sqrt(mean((yhat .- y).^2))
r2(yhat, y) = 1 .- sum((y .- yhat).^2) / max(sum((y .- mean(y)).^2), eps())

"""
Algorithm 4: UDE Conformal Calibration (from Structure-Preserving Models with Calibrated Uncertainty)

Steps:
1. For each val/test instance, predict \hat{u}_i = ODESolve(f*_θ, t_i), get y^ = \hat{u}_i[2]
2. Affine OLS calibration: a = sum((y^ - mean(y^)) * (y - mean(y))) / max(sum((y^ - mean(y^))^2), eps())
   b = mean(y) - a * mean(y^)
3. Residuals: r_i = |y_i - (a*y^_i + b)|; q_α = Quantile({r_i}, 1-α)
4. For test: calibrated prediction y~_j = a*\hat{u}_j[2] + b; intervals y~_j ± q_α
"""
function affine_calibration(yhat::Vector{Float64}, y::Vector{Float64})
    μyhat = mean(yhat); μy = mean(y)
    num = sum((yhat .- μyhat) .* (y .- μy))
    den = sum((yhat .- μyhat).^2)
    a = den == 0 ? 1.0 : num / max(den, eps())
    b = μy - a*μyhat
    return a, b
end

apply_affine(yhat, a, b) = a .* yhat .+ b

# Requested: conformal_quantile(abs_resid, alpha) = quantile(abs_resid, 1 - alpha)
conformal_quantile(abs_resid::Vector{Float64}, alpha::Float64) = quantile(abs_resid, 1 - alpha)

coverage(yhatp::Vector{Float64}, y::Vector{Float64}, q::Float64) =
    mean((y .>= yhatp .- q) .& (y .<= yhatp .+ q))

function gaussian_nll(yhatp::Vector{Float64}, y::Vector{Float64}, sigma::Float64)
    return mean(0.5*log(2*pi*sigma^2) .+ 0.5*((y .- yhatp).^2)./(sigma^2))
end

function must(b, msg)
    b || error(msg)
end

function main()
    path = get(ENV, "UDE_CAL_CSV", "results/ude_preds_x2.csv")
    must(isfile(path), "Missing predictions CSV: $(path)")
    df = CSV.read(path, DataFrame)
    nms = Set(String.(names(df)))
    must(all(s -> s in nms, ["split","yhat","y"]), "CSV must contain split,yhat,y")

    # split
    val = df[df.split .== "val", [:yhat, :y]]
    tst = df[df.split .== "test", [:yhat, :y]]
    must(nrow(val) > 5 && nrow(tst) > 5, "Insufficient rows in val/test")

    yhat_val = collect(Float64, val.yhat); y_val = collect(Float64, val.y)
    yhat_tst = collect(Float64, tst.yhat); y_tst = collect(Float64, tst.y)

    a,b = affine_calibration(yhat_val, y_val)
    yhatp_val = apply_affine(yhat_val, a, b)
    yhatp_tst = apply_affine(yhat_tst, a, b)

    abs_resid_val = abs.(y_val .- yhatp_val)
    q50 = conformal_quantile(abs_resid_val, 0.50)
    q90 = conformal_quantile(abs_resid_val, 0.10)

    metrics = Dict(
        "a" => a, "b" => b,
        "val" => Dict(
            "rmse_raw" => rmse(yhat_val, y_val),
            "r2_raw"   => r2(yhat_val, y_val),
            "rmse_cal" => rmse(yhatp_val, y_val),
            "r2_cal"   => r2(yhatp_val, y_val),
            "q50" => q50, "q90" => q90
        ),
        "test" => Dict(
            "rmse_raw" => rmse(yhat_tst, y_tst),
            "r2_raw"   => r2(yhat_tst, y_tst),
            "rmse_cal" => rmse(yhatp_tst, y_tst),
            "r2_cal"   => r2(yhatp_tst, y_tst),
            "coverage50" => coverage(yhatp_tst, y_tst, q50),
            "coverage90" => coverage(yhatp_tst, y_tst, q90),
            "width50_mean" => 2*q50,
            "width90_mean" => 2*q90
        )
    )

    sigma = std(y_val .- yhatp_val)
    metrics["test"]["nll_gaussian"] = gaussian_nll(yhatp_tst, y_tst, sigma)
    metrics["sigma_from_val"] = sigma

    mkpath("results"); mkpath("figures")
    JSON3.write("results/ude_calibration.json", merge(Dict("created_at" => string(now())), metrics))

    open("results/ude_calibration_table.csv", "w") do io
        println(io, "metric,raw,calibrated")
        @printf(io, "RMSE_test,%.6g,%.6g\n", metrics["test"]["rmse_raw"], metrics["test"]["rmse_cal"])
        @printf(io, "R2_test,%.6g,%.6g\n", metrics["test"]["r2_raw"], metrics["test"]["r2_cal"])
        @printf(io, "Coverage@50,,%.6g\n", metrics["test"]["coverage50"])
        @printf(io, "Coverage@90,,%.6g\n", metrics["test"]["coverage90"])
        @printf(io, "Width@50,,%.6g\n", metrics["test"]["width50_mean"])
        @printf(io, "Width@90,,%.6g\n", metrics["test"]["width90_mean"])
        @printf(io, "NLL_test,,%.6g\n", metrics["test"]["nll_gaussian"])
    end

    p = plot(1:length(y_tst), y_tst, label="y (test)", xlabel="Index (test)", ylabel="x₂")
    plot!(p, 1:length(y_tst), yhatp_tst, label="ŷ' (cal)")
    plot!(p, 1:length(y_tst), yhatp_tst .- q90, label="90% lo", ls=:dash)
    plot!(p, 1:length(y_tst), yhatp_tst .+ q90, label="90% hi", ls=:dash)
    plot!(p, 1:length(y_tst), yhatp_tst .- q50, label="50% lo", ls=:dot)
    plot!(p, 1:length(y_tst), yhatp_tst .+ q50, label="50% hi", ls=:dot)
    savefig(p, "figures/ude_calibration_intervals.pdf")

    println("UDE CALIBRATION COMPLETE")
end

main()


