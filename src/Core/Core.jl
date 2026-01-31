module Core

# ============================================================================
# Abstract Type Hierarchy
# ============================================================================

"""
    AbstractInstrument

Root type for all financial instruments.
"""
abstract type AbstractInstrument end

"""
    AbstractEquity <: AbstractInstrument

Abstract type for equity instruments (stocks, ETFs).
"""
abstract type AbstractEquity <: AbstractInstrument end

"""
    AbstractDerivative <: AbstractInstrument

Abstract type for derivative instruments.
"""
abstract type AbstractDerivative <: AbstractInstrument end

"""
    AbstractOption <: AbstractDerivative

Abstract type for option contracts.
"""
abstract type AbstractOption <: AbstractDerivative end

"""
    AbstractFuture <: AbstractDerivative

Abstract type for futures contracts.
"""
abstract type AbstractFuture <: AbstractDerivative end

"""
    AbstractPortfolio

Abstract type for portfolio containers.
"""
abstract type AbstractPortfolio end

"""
    AbstractRiskMeasure

Abstract type for risk measures (VaR, CVaR, etc.).
"""
abstract type AbstractRiskMeasure end

"""
    AbstractADBackend

Abstract type for automatic differentiation backends.
"""
abstract type AbstractADBackend end

# ============================================================================
# Core Types
# ============================================================================

# Simple immutable dict wrapper
struct ImmutableDict{K,V} <: AbstractDict{K,V}
    data::Dict{K,V}

    # Inner constructor that copies to ensure immutability
    function ImmutableDict{K,V}(d::Dict{K,V}) where {K,V}
        new{K,V}(copy(d))
    end
end
# Outer constructor for type inference
ImmutableDict(d::Dict{K,V}) where {K,V} = ImmutableDict{K,V}(d)

Base.getindex(d::ImmutableDict, k) = d.data[k]
Base.haskey(d::ImmutableDict, k) = haskey(d.data, k)
Base.keys(d::ImmutableDict) = keys(d.data)
Base.values(d::ImmutableDict) = values(d.data)
Base.length(d::ImmutableDict) = length(d.data)
Base.iterate(d::ImmutableDict, args...) = iterate(d.data, args...)
Base.setindex!(::ImmutableDict, v, k) = throw(MethodError(setindex!, (ImmutableDict, v, k)))

"""
    MarketState{P,R,V,T}

Immutable snapshot of market conditions.

# Fields
- `prices::P` - Current prices by symbol
- `rates::R` - Interest rates by currency
- `volatilities::V` - Implied volatilities by symbol
- `timestamp::T` - Time of snapshot
"""
struct MarketState{P,R,V,T}
    prices::P
    rates::R
    volatilities::V
    timestamp::T
end

# Keyword constructor for convenience
function MarketState(; prices, rates, volatilities, timestamp)
    # Convert to immutable dictionaries for safety
    MarketState(
        ImmutableDict(prices),
        ImmutableDict(rates),
        ImmutableDict(volatilities),
        timestamp
    )
end

# ============================================================================
# Exports
# ============================================================================

export AbstractInstrument, AbstractEquity, AbstractDerivative, AbstractOption, AbstractFuture
export AbstractPortfolio, AbstractRiskMeasure, AbstractADBackend
export MarketState, ImmutableDict

end
