module Calibration

using ..AD: gradient, ForwardDiffBackend
using ..Models: SABRParams, sabr_implied_vol, sabr_price, HestonParams, heston_price
using LinearAlgebra: norm
using Statistics: mean

# ============================================================================
# Market Data Types
# ============================================================================

"""
    OptionQuote

A single option quote from the market.

# Fields
- `strike::Float64` - Strike price
- `expiry::Float64` - Time to expiry (in years)
- `price::Float64` - Market price (can be 0 if only using implied_vol)
- `optiontype::Symbol` - :call or :put
- `implied_vol::Float64` - Market implied volatility
"""
struct OptionQuote
    strike::Float64
    expiry::Float64
    price::Float64
    optiontype::Symbol
    implied_vol::Float64
end

"""
    SmileData

Market data for a single expiry (a volatility smile).

# Fields
- `expiry::Float64` - Time to expiry (in years)
- `forward::Float64` - Forward price
- `rate::Float64` - Risk-free interest rate
- `quotes::Vector{OptionQuote}` - Option quotes at this expiry
"""
struct SmileData
    expiry::Float64
    forward::Float64
    rate::Float64
    quotes::Vector{OptionQuote}
end

# ============================================================================
# Calibration Results
# ============================================================================

"""
    CalibrationResult{P}

Result of model calibration.

# Fields
- `params::P` - Fitted model parameters
- `loss::Float64` - Final objective value
- `converged::Bool` - Whether optimization converged
- `iterations::Int` - Number of iterations taken
- `rmse::Float64` - Root mean squared error (in volatility terms)
"""
struct CalibrationResult{P}
    params::P
    loss::Float64
    converged::Bool
    iterations::Int
    rmse::Float64
end

# ============================================================================
# SABR Calibration
# ============================================================================

"""
    calibrate_sabr(smile::SmileData; beta=0.5, max_iter=1000, tol=1e-8, lr=0.01)

Calibrate SABR model to a single expiry volatility smile.

Uses gradient descent with automatic differentiation for optimization.
β is typically fixed (0.5 for rates, 1.0 for equities).

# Arguments
- `smile` - Market smile data for a single expiry
- `beta` - CEV exponent (fixed during calibration)
- `max_iter` - Maximum gradient descent iterations
- `tol` - Convergence tolerance on loss improvement
- `lr` - Learning rate

# Returns
CalibrationResult with fitted SABRParams.

# Example
```julia
quotes = [OptionQuote(K, 1.0, 0.0, :call, market_vol) for (K, market_vol) in market_data]
smile = SmileData(1.0, 100.0, 0.05, quotes)
result = calibrate_sabr(smile; beta=0.5)
```
"""
function calibrate_sabr(smile::SmileData;
                        beta::Float64=0.5,
                        max_iter::Int=1000,
                        tol::Float64=1e-8,
                        lr::Float64=0.01)

    F, T, r = smile.forward, smile.expiry, smile.rate
    quotes = smile.quotes
    n_quotes = length(quotes)

    # Find ATM quote for initial guess
    atm_idx = argmin([abs(q.strike - F) for q in quotes])
    atm_vol = quotes[atm_idx].implied_vol

    # Initial guess for α from ATM vol relation
    α_init = atm_vol / F^(beta - 1)

    # Estimate initial rho from skew (difference between low and high strike vols)
    low_strike_quotes = filter(q -> q.strike < F * 0.95, quotes)
    high_strike_quotes = filter(q -> q.strike > F * 1.05, quotes)
    ρ_init = 0.0
    if !isempty(low_strike_quotes) && !isempty(high_strike_quotes)
        low_vol = mean(q.implied_vol for q in low_strike_quotes)
        high_vol = mean(q.implied_vol for q in high_strike_quotes)
        # Negative skew (low > high) suggests negative rho
        skew = (low_vol - high_vol) / atm_vol
        ρ_init = clamp(-skew * 2, -0.8, 0.8)  # Heuristic mapping
    end

    # Pack free parameters: [α, ρ_unbounded, log_ν]
    # Transformations ensure constraints:
    #   α > 0 via abs()
    #   -1 < ρ < 1 via tanh()
    #   ν > 0 via exp()
    x = [α_init, atanh(clamp(ρ_init, -0.99, 0.99)), log(0.3)]

    # Loss function: mean squared implied vol error
    function loss(params)
        α = abs(params[1])           # α > 0
        ρ = tanh(params[2])          # -1 < ρ < 1
        ν = exp(params[3])           # ν > 0

        sabr = SABRParams(α, beta, ρ, ν)

        total_sq_error = zero(eltype(params))
        for q in quotes
            model_vol = sabr_implied_vol(F, q.strike, T, sabr)
            total_sq_error += (model_vol - q.implied_vol)^2
        end
        return total_sq_error / n_quotes
    end

    # Gradient descent with AD
    converged = false
    iter = 0
    prev_loss = Inf

    for i in 1:max_iter
        iter = i

        # Compute gradient using AD
        g = gradient(loss, x; backend=ForwardDiffBackend())

        # Update with gradient descent
        x_new = x - lr * g

        current_loss = loss(x_new)

        # Check convergence
        if abs(current_loss - prev_loss) < tol
            converged = true
            x = x_new
            break
        end

        # Adaptive learning rate: reduce if loss increased
        if current_loss > prev_loss * 1.1
            lr *= 0.5
        end

        x = x_new
        prev_loss = current_loss
    end

    # Extract final parameters
    α = abs(x[1])
    ρ = tanh(x[2])
    ν = exp(x[3])
    final_params = SABRParams(α, beta, ρ, ν)

    # Compute RMSE
    final_loss = loss(x)
    rmse = sqrt(final_loss)

    return CalibrationResult(final_params, final_loss, converged, iter, rmse)
end

# ============================================================================
# Heston Calibration
# ============================================================================

"""
    VolSurface

Market data for multiple expiries (full volatility surface).

# Fields
- `spot::Float64` - Current spot price
- `rate::Float64` - Risk-free interest rate
- `smiles::Vector{SmileData}` - Smile data for each expiry
"""
struct VolSurface
    spot::Float64
    rate::Float64
    smiles::Vector{SmileData}
end

"""
    calibrate_heston(surface::VolSurface; max_iter=2000, tol=1e-8, lr=0.001)

Calibrate Heston model to a full volatility surface.

Uses gradient descent with automatic differentiation. Fits a single set of
Heston parameters across all expiries in the surface.

# Arguments
- `surface` - Market volatility surface data
- `max_iter` - Maximum gradient descent iterations
- `tol` - Convergence tolerance on loss improvement
- `lr` - Learning rate

# Returns
CalibrationResult with fitted HestonParams.

# Example
```julia
smiles = [SmileData(T, F, r, quotes) for (T, F, quotes) in market_data]
surface = VolSurface(100.0, 0.05, smiles)
result = calibrate_heston(surface)
```
"""
function calibrate_heston(surface::VolSurface;
                          max_iter::Int=2000,
                          tol::Float64=1e-8,
                          lr::Float64=0.001)

    S, r = surface.spot, surface.rate
    smiles = surface.smiles

    # Count total quotes for normalization
    n_quotes = sum(length(smile.quotes) for smile in smiles)

    # Initial guess from ATM volatility
    # Find average ATM vol across expiries
    avg_atm_vol = 0.0
    for smile in smiles
        atm_idx = argmin([abs(q.strike - smile.forward) for q in smile.quotes])
        avg_atm_vol += smile.quotes[atm_idx].implied_vol
    end
    avg_atm_vol /= length(smiles)

    # Initial parameters: v0 ≈ σ_ATM², θ ≈ v0, κ = 1, σ = 0.3, ρ = -0.5
    v0_init = avg_atm_vol^2

    # Pack free parameters with transformations:
    # [log(v0), log(θ), log(κ), log(σ), atanh(ρ)]
    x = [log(v0_init), log(v0_init), log(1.0), log(0.3), atanh(-0.5)]

    # Loss function: mean squared price error (normalized by forward)
    function loss(params)
        v0 = exp(params[1])
        θ = exp(params[2])
        κ = exp(params[3])
        σ = exp(params[4])
        ρ = tanh(params[5])

        heston = HestonParams(v0, θ, κ, σ, ρ)

        total_sq_error = zero(eltype(params))
        for smile in smiles
            T = smile.expiry
            for q in smile.quotes
                model_price = heston_price(S, q.strike, T, r, heston, q.optiontype)
                # Normalize error by strike for numerical stability
                rel_error = (model_price - q.price) / q.strike
                total_sq_error += rel_error^2
            end
        end
        return total_sq_error / n_quotes
    end

    # Gradient descent with AD
    converged = false
    iter = 0
    prev_loss = Inf
    current_lr = lr

    for i in 1:max_iter
        iter = i

        # Compute gradient using AD
        g = gradient(loss, x; backend=ForwardDiffBackend())

        # Gradient clipping for stability
        g_norm = norm(g)
        if g_norm > 10.0
            g = g * (10.0 / g_norm)
        end

        # Update with gradient descent
        x_new = x - current_lr * g

        current_loss = loss(x_new)

        # Check convergence
        if abs(current_loss - prev_loss) < tol
            converged = true
            x = x_new
            break
        end

        # Adaptive learning rate
        if current_loss > prev_loss * 1.1
            current_lr *= 0.5
        elseif current_loss < prev_loss * 0.99
            current_lr = min(current_lr * 1.05, lr * 2)
        end

        x = x_new
        prev_loss = current_loss
    end

    # Extract final parameters
    v0 = exp(x[1])
    θ = exp(x[2])
    κ = exp(x[3])
    σ = exp(x[4])
    ρ = tanh(x[5])
    final_params = HestonParams(v0, θ, κ, σ, ρ)

    # Compute RMSE
    final_loss = loss(x)
    rmse = sqrt(final_loss)

    return CalibrationResult(final_params, final_loss, converged, iter, rmse)
end

# ============================================================================
# Exports
# ============================================================================

export OptionQuote, SmileData, CalibrationResult, calibrate_sabr
export VolSurface, calibrate_heston

end
