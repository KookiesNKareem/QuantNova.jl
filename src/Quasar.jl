module Quasar

# Core must come first - defines abstract types
include("Core.jl")
using .Core
export AbstractInstrument, AbstractEquity, AbstractDerivative, AbstractOption, AbstractFuture
export AbstractPortfolio, AbstractRiskMeasure, AbstractADBackend
export MarketState
export Priceable, IsPriceable, NotPriceable
export Differentiable, IsDifferentiable, NotDifferentiable
export HasGreeks, HasGreeksTrait, NoGreeksTrait
export Simulatable, IsSimulatable, NotSimulatable
export priceable, ispriceable, differentiable, isdifferentiable
export greeks_trait, hasgreeks, simulatable, issimulatable

# AD backend system
include("AD.jl")
using .AD
export PureJuliaBackend, ForwardDiffBackend, ReactantBackend
export gradient, hessian, jacobian, current_backend, set_backend!

# Instruments
include("Instruments.jl")
using .Instruments
export Stock, EuropeanOption, AmericanOption, AsianOption, price, black_scholes
export GreeksResult, compute_greeks

# Portfolio
include("Portfolio.jl")
using .PortfolioModule
export Portfolio, value, portfolio_greeks

# Risk measures
include("Risk.jl")
using .Risk
export VaR, CVaR, Volatility, Sharpe, MaxDrawdown, compute

# Optimization
include("Optimization.jl")
using .Optimization
export optimize, MeanVariance, SharpeMaximizer, CVaRObjective, KellyCriterion, OptimizationResult

# Stochastic volatility models (SABR, Heston)
include("Models.jl")
using .Models
export SABRParams, sabr_implied_vol, sabr_price, black76
export HestonParams, heston_price, heston_characteristic

# Model calibration
include("Calibration.jl")
using .Calibration
export OptionQuote, SmileData, VolSurface, CalibrationResult
export calibrate_sabr, calibrate_heston

# Market data
include("MarketData.jl")
using .MarketData
export AbstractMarketData, AbstractPriceHistory, CSVAdapter, ParquetAdapter

end
