module Backtesting

# TODO: Add MomentumStrategy for trend-following
# TODO: Add MeanReversionStrategy for contrarian trades
# TODO: Add CompositeStrategy for combining multiple strategies

using Dates
using Statistics: mean, std
using ..Simulation: SimulationState, Order, Fill, portfolio_value
using ..Simulation: AbstractDriver, HistoricalDriver, MarketSnapshot
using ..Simulation: AbstractExecutionModel, InstantFill, execute

# ============================================================================
# Strategy Interface
# ============================================================================

abstract type AbstractStrategy end

"""
    generate_orders(strategy, state) -> Vector{Order}

Generate orders based on strategy logic and current state.
"""
function generate_orders end

"""
    should_rebalance(strategy, state) -> Bool

Check if strategy should rebalance at current state.
"""
should_rebalance(::AbstractStrategy, ::SimulationState) = false

# ============================================================================
# Buy and Hold Strategy
# ============================================================================

"""
    BuyAndHoldStrategy <: AbstractStrategy

Invest in target weights once and hold.

# Fields
- `target_weights::Dict{Symbol,Float64}` - Target allocation (must sum to 1.0)
- `invested::Base.RefValue{Bool}` - Track if initial investment made

# Example
```julia
strategy = BuyAndHoldStrategy(Dict(:AAPL => 0.6, :GOOGL => 0.4))
orders = generate_orders(strategy, state)
```
"""
struct BuyAndHoldStrategy <: AbstractStrategy
    target_weights::Dict{Symbol,Float64}
    invested::Base.RefValue{Bool}

    function BuyAndHoldStrategy(target_weights::Dict{Symbol,Float64})
        total = sum(values(target_weights))
        abs(total - 1.0) < 0.01 || error("Target weights must sum to 1.0, got $total")
        new(target_weights, Ref(false))
    end
end

function generate_orders(strategy::BuyAndHoldStrategy, state::SimulationState)
    # Only invest once
    strategy.invested[] && return Order[]

    orders = Order[]
    total_value = portfolio_value(state)

    for (sym, target_weight) in strategy.target_weights
        haskey(state.prices, sym) || continue

        target_value = total_value * target_weight
        current_value = get(state.positions, sym, 0.0) * state.prices[sym]
        diff_value = target_value - current_value

        if abs(diff_value) > 1.0  # Minimum trade threshold
            qty = diff_value / state.prices[sym]
            side = qty > 0 ? :buy : :sell
            push!(orders, Order(sym, abs(qty), side))
        end
    end

    strategy.invested[] = true
    return orders
end

# ============================================================================
# Rebalancing Strategy
# ============================================================================

"""
    RebalancingStrategy <: AbstractStrategy

Periodically rebalance to target weights.

# Fields
- `target_weights::Dict{Symbol,Float64}` - Target allocation (must sum to 1.0)
- `rebalance_frequency::Symbol` - One of :daily, :weekly, :monthly
- `tolerance::Float64` - Rebalance if off by more than this fraction
- `last_rebalance::Base.RefValue{Union{Nothing,DateTime}}` - Last rebalance time

# Example
```julia
strategy = RebalancingStrategy(
    target_weights=Dict(:AAPL => 0.5, :GOOGL => 0.5),
    rebalance_frequency=:monthly,
    tolerance=0.05
)
```
"""
struct RebalancingStrategy <: AbstractStrategy
    target_weights::Dict{Symbol,Float64}
    rebalance_frequency::Symbol  # :daily, :weekly, :monthly
    tolerance::Float64           # Rebalance if off by more than this
    last_rebalance::Base.RefValue{Union{Nothing,DateTime}}

    function RebalancingStrategy(;
        target_weights::Dict{Symbol,Float64},
        rebalance_frequency::Symbol=:monthly,
        tolerance::Float64=0.05
    )
        total = sum(values(target_weights))
        abs(total - 1.0) < 0.01 || error("Target weights must sum to 1.0")
        rebalance_frequency in (:daily, :weekly, :monthly) ||
            error("rebalance_frequency must be :daily, :weekly, or :monthly")
        new(target_weights, rebalance_frequency, tolerance, Ref{Union{Nothing,DateTime}}(nothing))
    end
end

function should_rebalance(strategy::RebalancingStrategy, state::SimulationState)
    # Check time-based trigger
    if !isnothing(strategy.last_rebalance[])
        last = strategy.last_rebalance[]
        current = state.timestamp

        should_by_time = if strategy.rebalance_frequency == :daily
            Date(current) > Date(last)
        elseif strategy.rebalance_frequency == :weekly
            week(current) != week(last) || year(current) != year(last)
        else  # monthly
            month(current) != month(last) || year(current) != year(last)
        end

        !should_by_time && return false
    end

    # Check if weights are off target
    total_value = portfolio_value(state)
    total_value < 1.0 && return false

    for (sym, target) in strategy.target_weights
        current_value = get(state.positions, sym, 0.0) * get(state.prices, sym, 0.0)
        current_weight = current_value / total_value
        if abs(current_weight - target) > strategy.tolerance
            return true
        end
    end

    return false
end

function generate_orders(strategy::RebalancingStrategy, state::SimulationState)
    should_rebalance(strategy, state) || return Order[]

    orders = Order[]
    total_value = portfolio_value(state)

    for (sym, target_weight) in strategy.target_weights
        haskey(state.prices, sym) || continue

        target_value = total_value * target_weight
        current_value = get(state.positions, sym, 0.0) * state.prices[sym]
        diff_value = target_value - current_value

        if abs(diff_value) > 1.0
            qty = diff_value / state.prices[sym]
            side = qty > 0 ? :buy : :sell
            push!(orders, Order(sym, abs(qty), side))
        end
    end

    strategy.last_rebalance[] = state.timestamp
    return orders
end

# ============================================================================
# Backtest Result
# ============================================================================

"""
    BacktestResult

Complete results from running a backtest.
"""
struct BacktestResult
    initial_value::Float64
    final_value::Float64
    equity_curve::Vector{Float64}
    returns::Vector{Float64}
    timestamps::Vector{DateTime}
    trades::Vector{Fill}
    positions_history::Vector{Dict{Symbol,Float64}}
    metrics::Dict{Symbol,Float64}
end

# ============================================================================
# Backtest Runner
# ============================================================================

"""
    backtest(strategy, timestamps, price_series; kwargs...)

Run a full backtest simulation.

# Arguments
- `strategy::AbstractStrategy` - Trading strategy
- `timestamps::Vector{DateTime}` - Time series dates
- `price_series::Dict{Symbol,Vector{Float64}}` - Price data per asset
- `initial_cash::Float64=100_000.0` - Starting capital
- `execution_model::AbstractExecutionModel=InstantFill()` - How orders execute

# Returns
`BacktestResult` with equity curve, trades, and performance metrics.
"""
function backtest(
    strategy::AbstractStrategy,
    timestamps::Vector{DateTime},
    price_series::Dict{Symbol,Vector{Float64}};
    initial_cash::Float64=100_000.0,
    execution_model::AbstractExecutionModel=InstantFill()
)
    n = length(timestamps)

    # Initialize
    positions = Dict{Symbol,Float64}()
    cash = initial_cash

    equity_curve = Float64[]
    returns_vec = Float64[]
    positions_history = Dict{Symbol,Float64}[]
    all_trades = Fill[]

    prev_value = initial_cash

    for i in 1:n
        # Get current prices
        prices = Dict{Symbol,Float64}()
        for (sym, series) in price_series
            prices[sym] = series[i]
        end

        # Create current state
        state = SimulationState(
            timestamp=timestamps[i],
            cash=cash,
            positions=copy(positions),
            prices=prices
        )

        # Generate and execute orders
        orders = generate_orders(strategy, state)
        for order in orders
            fill = execute(execution_model, order, prices; timestamp=timestamps[i])
            push!(all_trades, fill)

            # Update positions and cash
            sym = fill.symbol
            positions[sym] = get(positions, sym, 0.0) + fill.quantity
            cash -= fill.quantity > 0 ? fill.cost : -fill.cost
        end

        # Record equity
        current_value = cash
        for (sym, qty) in positions
            current_value += qty * prices[sym]
        end
        push!(equity_curve, current_value)
        push!(positions_history, copy(positions))

        # Compute return
        if i > 1
            ret = (current_value - prev_value) / prev_value
            push!(returns_vec, ret)
        end
        prev_value = current_value
    end

    # Compute metrics
    metrics = compute_backtest_metrics(equity_curve, returns_vec)

    BacktestResult(
        initial_cash,
        equity_curve[end],
        equity_curve,
        returns_vec,
        timestamps,
        all_trades,
        positions_history,
        metrics
    )
end

"""
    compute_backtest_metrics(equity_curve, returns)

Compute standard backtest performance metrics.
"""
function compute_backtest_metrics(equity_curve::Vector{Float64}, returns::Vector{Float64})
    metrics = Dict{Symbol,Float64}()

    # Total return
    metrics[:total_return] = (equity_curve[end] - equity_curve[1]) / equity_curve[1]

    # Annualized return (assuming daily data, 252 trading days)
    n_periods = length(returns)
    if n_periods > 0
        metrics[:annualized_return] = (1 + metrics[:total_return])^(252 / n_periods) - 1
    else
        metrics[:annualized_return] = 0.0
    end

    # Volatility (annualized)
    if length(returns) > 1
        metrics[:volatility] = std(returns) * sqrt(252)
    else
        metrics[:volatility] = 0.0
    end

    # Sharpe ratio (assuming 0 risk-free rate)
    if metrics[:volatility] > 0
        metrics[:sharpe_ratio] = metrics[:annualized_return] / metrics[:volatility]
    else
        metrics[:sharpe_ratio] = 0.0
    end

    # Max drawdown
    peak = equity_curve[1]
    max_dd = 0.0
    for v in equity_curve
        peak = max(peak, v)
        dd = (peak - v) / peak
        max_dd = max(max_dd, dd)
    end
    metrics[:max_drawdown] = -max_dd

    # Calmar ratio
    if max_dd > 0
        metrics[:calmar_ratio] = metrics[:annualized_return] / max_dd
    else
        metrics[:calmar_ratio] = 0.0
    end

    # Win rate (if there are returns)
    if !isempty(returns)
        metrics[:win_rate] = count(r -> r > 0, returns) / length(returns)
    else
        metrics[:win_rate] = 0.0
    end

    return metrics
end

# ============================================================================
# Exports
# ============================================================================

export AbstractStrategy, generate_orders, should_rebalance
export BuyAndHoldStrategy, RebalancingStrategy
export BacktestResult, backtest, compute_backtest_metrics

end
