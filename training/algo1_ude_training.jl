#!/usr/bin/env julia

"""
Algorithm 1: UDE Training Pipeline (from Structure-Preserving Models with Calibrated Uncertainty)

Steps:
1. Grid search over (width w, λ, η, reltol, seed) as in Table 2
2. For each configuration:
   a. Initialize θ ~ N(0, 0.1²I) with seed s
   b. Set patience ← 0, epoch ← 0
   c. while epoch < 1000 and patience < 50:
      i.   Solve ODE using reltol
      ii.  Compute loss: L = RMSEₓ₂ + 0.2·RMSEₓ₁ + 0.1·MAPEₓ₂ + λ∥θ∥₂²
      iii. Update θ ← θ − η∇θL using L-BFGS
      iv.  Every 5 epochs: compute validation loss; reset patience only if validation loss improves
      v.   epoch += 1
   d. If best model found, record θ* and loss L*
"""

"""
corrected_ude_tuning.jl

UDE training with loss tracking and hyperparameter optimization.
"""

using Pkg
Pkg.activate(".")

using DifferentialEquations, Optim, Random, Statistics, LinearAlgebra
using CSV, DataFrames, BSON, Dates, JSON3
using Plots, StatsPlots
using Printf

# Include checkpoint manager
include("checkpoint_manager.jl")
using .CheckpointManager

# Configuration
const CONFIG = Dict(
    "data_paths" => Dict(
        "training" => "data/training_roadmap.csv",
        "validation" => "data/validation_roadmap.csv",
        "test" => "data/test_roadmap.csv"
    ),
    "output_paths" => Dict(
        "checkpoints" => "checkpoints/ude",
        "loss_curves" => "results/loss_curves",
        "hyperparameter_results" => "results/hyperparameter_search.csv"
    ),
    "training" => Dict(
        "max_epochs" => 1000,
        "patience" => 50,  # Early stopping patience
        "checkpoint_frequency" => 10,  # Save checkpoint every N epochs
        "validation_frequency" => 5,   # Validate every N epochs
        "loss_tracking_frequency" => 1  # Track loss every epoch
    ),
    "hyperparameter_search" => Dict(
        "widths" => [3, 4, 5, 6, 8, 10],
        "lambdas" => [1e-6, 1e-5, 1e-4, 5e-4, 1e-3, 5e-3],
        "learning_rates" => [1e-3, 5e-3, 1e-2, 5e-2],
        "reltols" => [1e-4, 1e-5, 1e-6, 1e-7],
        "seeds" => collect(0:9)
    )
)

"""
    struct TrainingMetrics

Container for training metrics and loss tracking.
"""
mutable struct TrainingMetrics
    epochs::Vector{Int}
    train_losses::Vector{Float64}
    val_losses::Vector{Float64}
    train_rmse_x1::Vector{Float64}
    train_rmse_x2::Vector{Float64}
    train_mape_x2::Vector{Float64}
    val_rmse_x1::Vector{Float64}
    val_rmse_x2::Vector{Float64}
    val_mape_x2::Vector{Float64}
    learning_rates::Vector{Float64}
    gradient_norms::Vector{Float64}
    convergence_metrics::Dict{String, Any}
    
    function TrainingMetrics()
        new(
            Int[], Float64[], Float64[],
            Float64[], Float64[], Float64[],
            Float64[], Float64[], Float64[],
            Float64[], Float64[],
            Dict{String, Any}()
        )
    end
end

"""
    load_microgrid_data(data_path::String) -> DataFrame

Load microgrid training/validation data from CSV.
"""
function load_microgrid_data(data_path::String)::DataFrame
    if !isfile(data_path)
        error("Data file not found: $data_path")
    end
    
    df = CSV.read(data_path, DataFrame)
    println("Loaded data: $(nrow(df)) rows, $(length(unique(df.scenario))) scenarios")
    
    return df
end

"""
    build_ude_theta(width::Int) -> Vector{Float64}

Build UDE neural network parameters for given width.
"""
function build_ude_theta(width::Int)::Vector{Float64}
    # Neural network for residual fθ(Pgen): 1D input only
    # Parameters: W1 ∈ R^{width×1}, b1 ∈ R^{width}, w2 ∈ R^{width}, b2 ∈ R
    n_inputs = 1
    W1 = randn(width, n_inputs) * 0.1
    b1 = randn(width) * 0.1
    w2 = randn(width) * 0.1
    b2 = randn() * 0.1
    return vcat(vec(W1), b1, w2, b2)
end

"""
    ude_residual!(du, u, p, t)

UDE residual function for microgrid dynamics.
"""
function ude_residual!(du, u, p, t)
    x1, x2 = u
    θ, width, ηin, ηout, α, γ, β, Pgen_vec, Pload_vec, d_vec, t_vec = p
    
    # Control input (simplified)
    u_control = 0.0
    
    # Interpolate exogenous signals at current time
    Pgen = lininterp(t_vec, Pgen_vec, t)
    Pload = lininterp(t_vec, Pload_vec, t)
    d = lininterp(t_vec, d_vec, t)

    # Storage dynamics (known physics)
    du[1] = ηin * u_control * (u_control > 0) - (1/ηout) * u_control * (u_control < 0) - d
    
    # Grid flow dynamics with UDE residual
    # Paper Eq. 4: dx₂/dt = −α·x₂ + fθ(Pgen) − β·Pload + γ·x₁
    f_theta = evaluate_neural_network([Pgen], θ, width)
    du[2] = -α * x2 + f_theta - β * Pload + γ * x1
end

"""
    evaluate_neural_network(inputs::Vector{Float64}, θ::Vector{Float64}, width::Int) -> Float64

Evaluate neural network for UDE residual.
"""
function evaluate_neural_network(inputs::Vector{Float64}, θ::Vector{Float64}, width::Int)::Float64
    # Unpack 1D-input residual network parameters
    n_inputs = 1
    idx = 1
    W1 = reshape(θ[idx:idx+width*n_inputs-1], width, n_inputs); idx += width*n_inputs
    b1 = θ[idx:idx+width-1]; idx += width
    w2 = θ[idx:idx+width-1]; idx += width
    b2 = θ[idx]
    h = tanh.(W1 * inputs .+ b1)
    return dot(w2, h) + b2
end

"""
    total_loss_corrected(params::Vector{Float64}, width::Int, reltol::Float64, 
                        scenarios::DataFrame, λ::Float64) -> Float64

Multi-objective loss function: RMSE + MAPE penalty + L2 regularization.
"""
function total_loss_corrected(params::Vector{Float64}, width::Int, reltol::Float64, 
                             scenarios::DataFrame, λ::Float64)::Float64
    
    total_loss = 0.0
    total_weight = 0.0
    
    for scenario in unique(scenarios.scenario)
        scenario_data = scenarios[scenarios.scenario .== scenario, :]
        
        # Extract scenario parameters
        ηin = scenario_data.ηin[1]
        ηout = scenario_data.ηout[1]
        α = scenario_data.α[1]
        γ = scenario_data.γ[1]
        β = scenario_data.β[1]
        
        # Time points
        t_span = (scenario_data.t[1], scenario_data.t[end])
        t_points = scenario_data.t
        
        # Initial conditions
        u0 = [scenario_data.x1[1], scenario_data.x2[1]]
        
        # Solve ODE
        try
            prob = ODEProblem(ude_residual!, u0, t_span, (params, width, ηin, ηout, α, γ, β, 
                                                         scenario_data.Pgen, scenario_data.Pload, 
                                                         scenario_data.d, scenario_data.t))
            sol = solve(prob, Tsit5(); saveat=t_points, reltol=reltol, abstol=reltol)
            
            if sol.retcode == :Success
                # Compute metrics
                pred_x1 = [u[1] for u in sol.u]
                pred_x2 = [u[2] for u in sol.u]
                true_x1 = scenario_data.x1
                true_x2 = scenario_data.x2
                
                # RMSE components
                rmse_x1 = sqrt(mean((pred_x1 .- true_x1).^2))
                rmse_x2 = sqrt(mean((pred_x2 .- true_x2).^2))
                
                # MAPE for x2 (frequency)
                mape_x2 = mean(abs.((pred_x2 .- true_x2) ./ (true_x2 .+ 1e-8)))
                
                # Multi-objective loss
                scenario_loss = rmse_x2 + 0.2 * rmse_x1 + 0.1 * mape_x2
                total_loss += scenario_loss
                total_weight += 1.0
            else
                # Penalty for failed solves
                total_loss += 1000.0
                total_weight += 1.0
            end
        catch
            # Penalty for errors
            total_loss += 1000.0
            total_weight += 1.0
        end
    end
    
    # Average loss
    avg_loss = total_weight > 0 ? total_loss / total_weight : 1000.0
    
    # L2 regularization
    l2_penalty = λ * sum(params.^2)
    
    return avg_loss + l2_penalty
end

"""
    compute_validation_metrics(params::Vector{Float64}, width::Int, reltol::Float64, 
                              val_scenarios::DataFrame) -> Dict{String, Float64}

Compute validation metrics for monitoring.
"""
function compute_validation_metrics(params::Vector{Float64}, width::Int, reltol::Float64, 
                                   val_scenarios::DataFrame)::Dict{String, Float64}
    
    metrics = Dict{String, Float64}()
    total_rmse_x1 = 0.0
    total_rmse_x2 = 0.0
    total_mape_x2 = 0.0
    total_weight = 0.0
    
    for scenario in unique(val_scenarios.scenario)
        scenario_data = val_scenarios[val_scenarios.scenario .== scenario, :]
        
        # Extract scenario parameters
        ηin = scenario_data.ηin[1]
        ηout = scenario_data.ηout[1]
        α = scenario_data.α[1]
        γ = scenario_data.γ[1]
        β = scenario_data.β[1]
        
        # Time points
        t_span = (scenario_data.t[1], scenario_data.t[end])
        t_points = scenario_data.t
        
        # Initial conditions
        u0 = [scenario_data.x1[1], scenario_data.x2[1]]
        
        # Solve ODE
        try
            prob = ODEProblem(ude_residual!, u0, t_span, (params, width, ηin, ηout, α, γ, β, 
                                                         scenario_data.Pgen, scenario_data.Pload, 
                                                         scenario_data.d, scenario_data.t))
            sol = solve(prob, Tsit5(); saveat=t_points, reltol=reltol, abstol=reltol)
            
            if sol.retcode == :Success
                # Compute metrics
                pred_x1 = [u[1] for u in sol.u]
                pred_x2 = [u[2] for u in sol.u]
                true_x1 = scenario_data.x1
                true_x2 = scenario_data.x2
                
                # RMSE components
                rmse_x1 = sqrt(mean((pred_x1 .- true_x1).^2))
                rmse_x2 = sqrt(mean((pred_x2 .- true_x2).^2))
                
                # MAPE for x2
                mape_x2 = mean(abs.((pred_x2 .- true_x2) ./ (true_x2 .+ 1e-8)))
                
                total_rmse_x1 += rmse_x1
                total_rmse_x2 += rmse_x2
                total_mape_x2 += mape_x2
                total_weight += 1.0
            end
        catch
            # Skip failed scenarios
        end
    end
    
    if total_weight > 0
        metrics["rmse_x1"] = total_rmse_x1 / total_weight
        metrics["rmse_x2"] = total_rmse_x2 / total_weight
        metrics["mape_x2"] = total_mape_x2 / total_weight
        metrics["val_loss"] = metrics["rmse_x2"] + 0.2 * metrics["rmse_x1"] + 0.1 * metrics["mape_x2"]
    else
        metrics["rmse_x1"] = Inf
        metrics["rmse_x2"] = Inf
        metrics["mape_x2"] = Inf
        metrics["val_loss"] = Inf
    end
    
    return metrics
end

"""
    optimize_corrected(width::Int, λ::Float64, lr::Float64, reltol::Float64, seed::Int;
                      train_data::DataFrame, val_data::DataFrame, 
                      max_epochs::Int=1000, patience::Int=50) -> Tuple

Enhanced UDE optimization with comprehensive loss tracking and early stopping.
"""
function optimize_corrected(width::Int, λ::Float64, reltol::Float64, seed::Int, lr::Float64;
                           train_data::DataFrame, val_data::DataFrame, 
                           max_epochs::Int=1000, patience::Int=50)::Tuple
    
    Random.seed!(seed)
    
    # Initialize parameters
    θ = build_ude_theta(width)
    n_params = length(θ)
    
    println("Starting UDE optimization...")
    println("Width: $width, λ: $λ, reltol: $reltol, seed: $seed, lr: $lr")
    println("Parameters: $n_params")
    
    # Initialize tracking
    metrics = TrainingMetrics()
    best_loss = Inf
    best_params = copy(θ)
    best_val_loss = Inf
    patience_counter = 0
    start_time = time()
    
    # Loss function wrapper
    function loss_wrapper(params)
        return total_loss_corrected(params, width, reltol, train_data, λ)
    end
    
    # Finite-difference gradient (simple but robust) for L-BFGS
    function finite_diff_grad(params::Vector{Float64})
        g = similar(params)
        base_loss = loss_wrapper(params)
        δ = 1e-6
        for i in eachindex(params)
            old = params[i]
            params[i] = old + δ
            li = loss_wrapper(params)
            g[i] = (li - base_loss) / δ
            params[i] = old
        end
        return g
    end

    # Optimization loop with L-BFGS
    for epoch in 1:max_epochs
        epoch_start = time()
        
        # Compute training loss
        train_loss = loss_wrapper(θ)
        
        # Compute validation metrics every N epochs
        val_metrics = Dict{String, Float64}()
        val_loss_epoch = nothing
        if epoch % CONFIG["training"]["validation_frequency"] == 0 || epoch == 1
            val_metrics = compute_validation_metrics(θ, width, reltol, val_data)
            if haskey(val_metrics, "val_loss")
                val_loss_epoch = val_metrics["val_loss"]
                push!(metrics.val_losses, val_loss_epoch)
                if val_loss_epoch < best_val_loss
                    best_val_loss = val_loss_epoch
                    patience_counter = 0
                else
                    patience_counter += 1
                end
            end
        else
            push!(metrics.val_losses, NaN)
            # Do not increment patience when not validating
        end
        
        # Track metrics
        push!(metrics.epochs, epoch)
        push!(metrics.train_losses, train_loss)
        
        if haskey(val_metrics, "val_loss")
            push!(metrics.val_rmse_x1, val_metrics["rmse_x1"])
            push!(metrics.val_rmse_x2, val_metrics["rmse_x2"])
            push!(metrics.val_mape_x2, val_metrics["mape_x2"])
        else
            push!(metrics.val_rmse_x1, NaN)
            push!(metrics.val_rmse_x2, NaN)
            push!(metrics.val_mape_x2, NaN)
        end
        
        # Compute additional metrics
        if epoch > 1
            # Gradient norm (approximate)
            grad_norm = norm(θ - best_params)
            push!(metrics.gradient_norms, grad_norm)
        else
            push!(metrics.gradient_norms, NaN)
        end
        
        push!(metrics.learning_rates, NaN)
        
        # Update best parameters by training loss (patience governed solely by validation)
        if train_loss < best_loss
            best_loss = train_loss
            best_params = copy(θ)
        end
        
        # Early stopping only after validation epochs
        if (epoch % CONFIG["training"]["validation_frequency"] == 0) && (patience_counter >= patience)
            println("Early stopping at epoch $epoch (patience: $patience)")
            break
        end

        # L-BFGS step (optimize current θ using loss and finite-diff gradient)
        function loss_and_grad!(F, G, params)
            loss_val = loss_wrapper(params)
            if G !== nothing
                G .= finite_diff_grad(params)
            end
            return loss_val
        end
        try
            res = optimize(Optim.only_fg!(loss_and_grad!), θ, LBFGS();
                           iterations=50, show_trace=false)
            θ = Optim.minimizer(res)
        catch e
            println("⚠️  L-BFGS step failed at epoch $epoch: $e")
        end
        
        # Logging
        epoch_time = time() - epoch_start
        if epoch % 10 == 0 || epoch == 1
            val_str = haskey(val_metrics, "val_loss") ? @sprintf("%.6f", val_metrics["val_loss"]) : "N/A"
            println("   Epoch $epoch: train_loss=$(round(train_loss, digits=6)), val_loss=$val_str, time=$(round(epoch_time, digits=2))s")
        end
        
        # Checkpointing
        if epoch % CONFIG["training"]["checkpoint_frequency"] == 0
            training_time = time() - start_time
            convergence_metrics = Dict(
                "patience_counter" => patience_counter,
                "best_loss" => best_loss,
                "current_loss" => train_loss,
                "gradient_norm" => metrics.gradient_norms[end]
            )
            
            checkpoint = create_checkpoint_metadata(
                "UDE", epoch, train_loss, 
                Dict("width" => width, "lambda" => λ, "lr" => lr, "reltol" => reltol, "seed" => seed),
                training_time;
                validation_loss = haskey(val_metrics, "val_loss") ? val_metrics["val_loss"] : nothing,
                model_parameters = copy(θ),
                convergence_metrics = convergence_metrics
            )
            
            save_checkpoint(checkpoint, CONFIG["output_paths"]["checkpoints"])
        end
    end
    
    total_time = time() - start_time
    println("Optimization complete!")
    println("Final loss: $(round(best_loss, digits=6))")
    println("Training time: $(round(total_time, digits=2))s")
    println("Epochs: $(length(metrics.epochs))")
    
    # Final convergence metrics
    metrics.convergence_metrics = Dict(
        "total_epochs" => length(metrics.epochs),
        "final_loss" => best_loss,
        "training_time" => total_time,
        "converged" => patience_counter < patience,
        "early_stopped" => patience_counter >= patience
    )
    
    return best_params, metrics, best_loss
end

"""
    hyperparameter_search(train_data::DataFrame, val_data::DataFrame) -> DataFrame

Perform comprehensive hyperparameter search with tracking.
"""
function hyperparameter_search(train_data::DataFrame, val_data::DataFrame)::DataFrame
    println("Starting hyperparameter search...")
    
    results = []
    total_configs = length(CONFIG["hyperparameter_search"]["widths"]) * 
                   length(CONFIG["hyperparameter_search"]["lambdas"]) * 
                   length(CONFIG["hyperparameter_search"]["reltols"]) * 
                   length(CONFIG["hyperparameter_search"]["seeds"]) 
    
    println("Total configurations: $total_configs")
    
    config_count = 0
    for width in CONFIG["hyperparameter_search"]["widths"]
        for λ in CONFIG["hyperparameter_search"]["lambdas"]
            for lr in CONFIG["hyperparameter_search"]["learning_rates"]
                for reltol in CONFIG["hyperparameter_search"]["reltols"]
                    for seed in CONFIG["hyperparameter_search"]["seeds"]
                        config_count += 1
                        
                        println("Configuration $config_count/$total_configs: width=$width, λ=$λ, lr=$lr, reltol=$reltol, seed=$seed")
                        
                        try
                            start_time = time()
                            params, metrics, final_loss = optimize_corrected(
                                width, λ, reltol, seed, lr;
                                train_data=train_data, val_data=val_data,
                                max_epochs=CONFIG["training"]["max_epochs"],
                                patience=CONFIG["training"]["patience"]
                            )
                            training_time = time() - start_time
                            
                            # Extract validation metrics
                            val_loss = length(metrics.val_losses) > 0 ? metrics.val_losses[end] : NaN
                            val_rmse_x2 = length(metrics.val_rmse_x2) > 0 ? metrics.val_rmse_x2[end] : NaN
                            
                            push!(results, Dict(
                                "width" => width,
                                "lambda" => λ,
                                "learning_rate" => lr,
                                "reltol" => reltol,
                                "seed" => seed,
                                "final_loss" => final_loss,
                                "val_loss" => val_loss,
                                "val_rmse_x2" => val_rmse_x2,
                                "training_time" => training_time,
                                "epochs" => length(metrics.epochs),
                                "converged" => metrics.convergence_metrics["converged"],
                                "early_stopped" => metrics.convergence_metrics["early_stopped"]
                            ))
                            
                            println("Loss: $(round(final_loss, digits=6)), Val: $(round(val_loss, digits=6)), Time: $(round(training_time, digits=1))s")
                            
                        catch e
                            println("Failed: $e")
                            push!(results, Dict(
                                "width" => width,
                                "lambda" => λ,
                                "reltol" => reltol,
                                "seed" => seed,
                                "final_loss" => Inf,
                                "val_loss" => Inf,
                                "val_rmse_x2" => Inf,
                                "training_time" => 0.0,
                                "epochs" => 0,
                                "converged" => false,
                                "early_stopped" => false,
                                "error" => string(e)
                            ))
                        end
                    end
                end
            end
        end
    end
    
    # Convert to DataFrame and save
    df = DataFrame(results)
    mkpath(dirname(CONFIG["output_paths"]["hyperparameter_results"]))
    CSV.write(CONFIG["output_paths"]["hyperparameter_results"], df)
    
    println("Hyperparameter search complete!")
    println("Results saved to: $(CONFIG["output_paths"]["hyperparameter_results"])")
    
    return df
end

"""
    generate_loss_curves(metrics::TrainingMetrics, output_dir::String)

Generate publication-quality loss curves.
"""
function generate_loss_curves(metrics::TrainingMetrics, output_dir::String)
    mkpath(output_dir)
    
    # Main loss curves
    p1 = plot(metrics.epochs, metrics.train_losses, 
              label="Training Loss", linewidth=2, color=:blue)
    plot!(p1, metrics.epochs[.!isnan.(metrics.val_losses)], 
          metrics.val_losses[.!isnan.(metrics.val_losses)], 
          label="Validation Loss", linewidth=2, color=:red, linestyle=:dash)
    plot!(p1, xlabel="Epoch", ylabel="Loss", title="UDE Training Loss Curves", 
          yscale=:log, grid=true, legend=:topright)
    
    # RMSE components
    p2 = plot(metrics.epochs[.!isnan.(metrics.val_rmse_x2)], 
              metrics.val_rmse_x2[.!isnan.(metrics.val_rmse_x2)], 
              label="RMSE x₂", linewidth=2, color=:green)
    plot!(p2, metrics.epochs[.!isnan.(metrics.val_rmse_x1)], 
          metrics.val_rmse_x1[.!isnan.(metrics.val_rmse_x1)], 
          label="RMSE x₁", linewidth=2, color=:orange)
    plot!(p2, xlabel="Epoch", ylabel="RMSE", title="Validation RMSE Components", 
          grid=true, legend=:topright)
    
    # Gradient norms
    p3 = plot(metrics.epochs[.!isnan.(metrics.gradient_norms)], 
              metrics.gradient_norms[.!isnan.(metrics.gradient_norms)], 
              label="Gradient Norm", linewidth=2, color=:purple)
    plot!(p3, xlabel="Epoch", ylabel="Gradient Norm", title="Gradient Norm Evolution", 
          yscale=:log, grid=true, legend=:topright)
    
    # Combined plot
    p_combined = plot(p1, p2, p3, layout=(3,1), size=(800, 1200))
    
    # Save plots
    savefig(p1, joinpath(output_dir, "loss_curves.pdf"))
    savefig(p2, joinpath(output_dir, "rmse_components.pdf"))
    savefig(p3, joinpath(output_dir, "gradient_norms.pdf"))
    savefig(p_combined, joinpath(output_dir, "training_summary.pdf"))
    
    println("Loss curves saved to: $output_dir")
end

"""
    main()

Main training pipeline.
"""
function main()
    println("UDE TRAINING WITH LOSS TRACKING")
    println("=" ^ 50)
    
    # Load data
    println("Loading data...")
    train_data = load_microgrid_data(CONFIG["data_paths"]["training"])
    val_data = load_microgrid_data(CONFIG["data_paths"]["validation"])
    
    # Create output directories
    mkpath(CONFIG["output_paths"]["checkpoints"])
    mkpath(CONFIG["output_paths"]["loss_curves"])
    
    # Run hyperparameter search
    results_df = hyperparameter_search(train_data, val_data)
    
    # Find best configuration
    best_idx = argmin(results_df.final_loss)
    best_config = results_df[best_idx, :]
    
    println("Best Configuration:")
    println("Width: $(best_config.width)")
    println("λ: $(best_config.lambda)")
    println("Learning Rate: $(best_config.learning_rate)")
    println("RelTol: $(best_config.reltol)")
    println("Seed: $(best_config.seed)")
    println("Final Loss: $(round(best_config.final_loss, digits=6))")
    
    # Train final model with best configuration
    println("Training final model...")
    final_params, final_metrics, final_loss = optimize_corrected(
        best_config.width, best_config.lambda, best_config.learning_rate, 
        best_config.reltol, best_config.seed;
        train_data=train_data, val_data=val_data
    )
    
    # Generate loss curves
    generate_loss_curves(final_metrics, CONFIG["output_paths"]["loss_curves"])
    
    # Save final model
    final_checkpoint = create_checkpoint_metadata(
        "UDE", length(final_metrics.epochs), final_loss,
        Dict("width" => best_config.width, "lambda" => best_config.lambda, 
             "lr" => best_config.learning_rate, "reltol" => best_config.reltol, 
             "seed" => best_config.seed),
        final_metrics.convergence_metrics["training_time"];
        model_parameters = final_params,
        convergence_metrics = final_metrics.convergence_metrics
    )
    
    final_path = save_checkpoint(final_checkpoint, CONFIG["output_paths"]["checkpoints"])
    println("Final model saved to: $final_path")
    
    println("Training complete!")
end

if abspath(PROGRAM_FILE) == @__FILE__
    main()
end

