module Models

using Distributions: Normal, cdf
using LinearAlgebra: eigen

# ============================================================================
# SABR Model
# ============================================================================

# SABR parameters: (α, β, ρ, ν)
# α = initial volatility level
# β = CEV exponent (usually fixed at 0.5 or 1.0)
# ρ = correlation between spot and vol (-1 to 1)
# ν = vol of vol

"""
    SABRParams{T1,T2,T3,T4}

SABR stochastic volatility model parameters.

# Fields
- `alpha::T1` - Initial volatility level (α > 0)
- `beta::T2` - CEV exponent (0 ≤ β ≤ 1, often fixed at 0.5 for rates, 1.0 for equities)
- `rho::T3` - Correlation between spot and vol (-1 < ρ < 1)
- `nu::T4` - Vol of vol (ν > 0)
"""
struct SABRParams{T1,T2,T3,T4}
    alpha::T1
    beta::T2
    rho::T3
    nu::T4
end

"""
    sabr_implied_vol(F, K, T, params::SABRParams)

Compute SABR implied volatility using Hagan's approximation formula.

# Arguments
- `F` - Forward price
- `K` - Strike price
- `T` - Time to expiry (in years)
- `params` - SABR model parameters

# Returns
Implied Black volatility for the given strike.
"""
function sabr_implied_vol(F, K, T, params::SABRParams)
    α, β, ρ, ν = params.alpha, params.beta, params.rho, params.nu

    # Handle ATM case separately to avoid numerical issues
    if abs(F - K) < 1e-12
        return _sabr_atm_vol(F, T, α, β, ρ, ν)
    end

    # Hagan's formula for non-ATM
    logFK = log(F / K)
    FK_mid = (F * K)^((1 - β) / 2)

    z = (ν / α) * FK_mid * logFK

    # Compute x(z) with care for small z
    if abs(z) < 1e-12
        x_z = 1.0
    else
        sqrt_term = sqrt(1 - 2*ρ*z + z^2)
        x_z = log((sqrt_term + z - ρ) / (1 - ρ))
        x_z = z / x_z
    end

    # Numerator: expansion in logFK
    denom_expansion = 1 + (1-β)^2/24 * logFK^2 + (1-β)^4/1920 * logFK^4
    A = α / (FK_mid * denom_expansion)

    # Correction terms for time
    C1 = ((1-β)^2 / 24) * (α^2 / FK_mid^2)
    C2 = (ρ * β * ν * α) / (4 * FK_mid)
    C3 = (2 - 3*ρ^2) * ν^2 / 24

    return A * x_z * (1 + (C1 + C2 + C3) * T)
end

"""
    _sabr_atm_vol(F, T, α, β, ρ, ν)

ATM SABR volatility (internal helper).
"""
function _sabr_atm_vol(F, T, α, β, ρ, ν)
    F_pow = F^(β - 1)
    C1 = ((1-β)^2 / 24) * α^2 * F_pow^2
    C2 = (ρ * β * ν * α * F_pow) / 4
    C3 = (2 - 3*ρ^2) * ν^2 / 24
    return α * F_pow * (1 + (C1 + C2 + C3) * T)
end

"""
    black76(F, K, T, r, σ, optiontype::Symbol)

Black-76 pricing formula for options on forwards/futures.

# Arguments
- `F` - Forward price
- `K` - Strike price
- `T` - Time to expiry (in years)
- `r` - Risk-free rate (for discounting)
- `σ` - Implied volatility
- `optiontype` - :call or :put

# Returns
Option price
"""
function black76(F, K, T, r, σ, optiontype::Symbol)
    if T <= 0 || σ <= 0
        # Handle edge cases
        if optiontype == :call
            return max(F - K, 0.0) * exp(-r * max(T, 0.0))
        else
            return max(K - F, 0.0) * exp(-r * max(T, 0.0))
        end
    end

    d1 = (log(F/K) + 0.5*σ^2*T) / (σ*sqrt(T))
    d2 = d1 - σ*sqrt(T)
    df = exp(-r*T)
    N = Normal()

    if optiontype == :call
        return df * (F * cdf(N, d1) - K * cdf(N, d2))
    else
        return df * (K * cdf(N, -d2) - F * cdf(N, -d1))
    end
end

"""
    sabr_price(F, K, T, r, params::SABRParams, optiontype::Symbol)

Price a European option using SABR implied vol + Black-76.

# Arguments
- `F` - Forward price
- `K` - Strike price
- `T` - Time to expiry (in years)
- `r` - Risk-free rate
- `params` - SABR model parameters
- `optiontype` - :call or :put

# Returns
Option price
"""
function sabr_price(F, K, T, r, params::SABRParams, optiontype::Symbol)
    σ = sabr_implied_vol(F, K, T, params)
    return black76(F, K, T, r, σ, optiontype)
end

# ============================================================================
# Heston Model
# ============================================================================

# Heston parameters: (v0, θ, κ, σ, ρ)
# v0 = initial variance
# θ = long-term variance (mean reversion level)
# κ = mean reversion speed
# σ = vol of vol
# ρ = correlation between spot and vol

"""
    HestonParams{T1,T2,T3,T4,T5}

Heston stochastic volatility model parameters.

# Fields
- `v0::T1` - Initial variance (v0 > 0)
- `theta::T2` - Long-term variance / mean reversion level (θ > 0)
- `kappa::T3` - Mean reversion speed (κ > 0)
- `sigma::T4` - Volatility of variance (σ > 0)
- `rho::T5` - Correlation between spot and variance (-1 < ρ < 1)

The Feller condition 2κθ > σ² ensures the variance stays positive.
"""
struct HestonParams{T1,T2,T3,T4,T5}
    v0::T1
    theta::T2
    kappa::T3
    sigma::T4
    rho::T5
end

"""
    heston_characteristic(u, S, K, T, r, params::HestonParams)

Compute the Heston characteristic function for pricing via Fourier methods.

Uses the log-stock formulation with the Gatheral parameterization for stability.
"""
function heston_characteristic(u, S, T, r, params::HestonParams)
    v0, θ, κ, σ, ρ = params.v0, params.theta, params.kappa, params.sigma, params.rho

    # Gatheral's notation (more stable)
    a = κ * θ
    b = κ

    # Complex intermediate values
    iu = im * u
    d = sqrt((ρ * σ * iu - b)^2 + σ^2 * (iu + u^2))
    g = (b - ρ * σ * iu - d) / (b - ρ * σ * iu + d)

    # Characteristic function components
    C = r * iu * T + (a / σ^2) * ((b - ρ * σ * iu - d) * T - 2 * log((1 - g * exp(-d * T)) / (1 - g)))
    D = ((b - ρ * σ * iu - d) / σ^2) * ((1 - exp(-d * T)) / (1 - g * exp(-d * T)))

    return exp(C + D * v0 + iu * log(S))
end

"""
    heston_price(S, K, T, r, params::HestonParams, optiontype::Symbol; N=128)

Price a European option under the Heston model using numerical integration.

# Arguments
- `S` - Spot price
- `K` - Strike price
- `T` - Time to expiry (in years)
- `r` - Risk-free rate
- `params` - Heston model parameters
- `optiontype` - :call or :put
- `N` - Number of integration points (default 128)

# Returns
Option price

# Notes
Uses the Gil-Pelaez / Carr-Madan approach with trapezoidal integration.
"""
function heston_price(S, K, T, r, params::HestonParams, optiontype::Symbol; N::Int=128)
    # Handle edge cases
    if T <= 0
        if optiontype == :call
            return max(S - K, 0.0)
        else
            return max(K - S, 0.0)
        end
    end

    # Gil-Pelaez inversion formula
    # P1 = 0.5 + (1/π) * ∫₀^∞ Re[exp(-iu*log(K)) * φ(u-i) / (iu * S * exp(rT))] du
    # P2 = 0.5 + (1/π) * ∫₀^∞ Re[exp(-iu*log(K)) * φ(u) / (iu)] du

    logK = log(K)
    logS = log(S)

    # Use trapezoidal rule with exponential decay for truncation
    # Integration limit and step size
    u_max = 100.0
    du = u_max / N

    P1 = 0.0
    P2 = 0.0

    for j in 1:N
        u = (j - 0.5) * du  # Midpoint rule

        # Characteristic function evaluations
        φ1 = heston_characteristic(u - im, S, T, r, params)
        φ2 = heston_characteristic(u, S, T, r, params)

        # Integrands for P1 and P2
        exp_term = exp(-im * u * logK)

        # P1 integrand: Re[exp(-iu*logK) * φ(u-i) / (iu * S * exp(rT))]
        integrand1 = real(exp_term * φ1 / (im * u * S * exp(r * T)))

        # P2 integrand: Re[exp(-iu*logK) * φ(u) / (iu)]
        integrand2 = real(exp_term * φ2 / (im * u))

        P1 += integrand1 * du
        P2 += integrand2 * du
    end

    P1 = 0.5 + P1 / π
    P2 = 0.5 + P2 / π

    # Ensure probabilities are in [0, 1]
    P1 = clamp(P1, 0.0, 1.0)
    P2 = clamp(P2, 0.0, 1.0)

    # Call price
    call_price = S * P1 - K * exp(-r * T) * P2

    if optiontype == :call
        return max(call_price, 0.0)
    else
        # Put-call parity
        put_price = call_price - S + K * exp(-r * T)
        return max(put_price, 0.0)
    end
end

# ============================================================================
# Exports
# ============================================================================

export SABRParams, sabr_implied_vol, sabr_price, black76
export HestonParams, heston_price, heston_characteristic

end
