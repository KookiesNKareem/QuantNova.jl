module ScenarioAnalysis

using Dates
using Statistics: mean, std, quantile
using ..Simulation: SimulationState, portfolio_value

# ============================================================================
# Stress Scenarios
# ============================================================================

"""
    StressScenario

Defines a stress scenario with shocks to different asset classes.

# Fields
- `name::String` - Human-readable scenario name
- `description::String` - Detailed description of the historical event
- `shocks::Dict{Symbol,Float64}` - Asset class to percent change mapping (e.g., -0.50 for -50%)
- `duration_days::Int` - Historical duration of the stress event
"""
struct StressScenario
    name::String
    description::String
    shocks::Dict{Symbol,Float64}  # asset_class => percent change (e.g., -0.50 for -50%)
    duration_days::Int
end

"""
    ScenarioImpact

Result of applying a stress scenario.

# Fields
- `scenario_name::String` - Name of the applied scenario
- `initial_value::Float64` - Portfolio value before stress
- `stressed_value::Float64` - Portfolio value after stress
- `pnl::Float64` - Profit/loss from the scenario
- `pct_change::Float64` - Percentage change in portfolio value
- `asset_impacts::Dict{Symbol,Float64}` - Per-asset P&L breakdown
"""
struct ScenarioImpact
    scenario_name::String
    initial_value::Float64
    stressed_value::Float64
    pnl::Float64
    pct_change::Float64
    asset_impacts::Dict{Symbol,Float64}
end

# ============================================================================
# Built-in Historical Crisis Scenarios
# ============================================================================

"""
    CRISIS_SCENARIOS

Dictionary of built-in historical crisis scenarios for stress testing.

Available scenarios:
- `:financial_crisis_2008` - Global financial crisis, S&P 500 fell ~57% from peak
- `:covid_crash_2020` - Rapid market crash in March 2020
- `:dot_com_bust_2000` - Tech bubble burst, NASDAQ fell ~78%
- `:black_monday_1987` - Single-day crash of 22.6%
- `:rate_shock_2022` - Fed aggressive tightening, bonds and stocks fell together
- `:stagflation_1970s` - High inflation with economic stagnation
"""
const CRISIS_SCENARIOS = Dict{Symbol,StressScenario}(
    :financial_crisis_2008 => StressScenario(
        "2008 Financial Crisis",
        "Global financial crisis - S&P 500 fell ~57% from peak",
        Dict(:equity => -0.50, :bond => 0.10, :commodity => -0.40, :gold => 0.05, :reit => -0.60),
        365
    ),
    :covid_crash_2020 => StressScenario(
        "COVID-19 Crash 2020",
        "Rapid market crash in March 2020 - S&P 500 fell ~34% in weeks",
        Dict(:equity => -0.34, :bond => 0.05, :commodity => -0.30, :gold => -0.05, :reit => -0.40),
        30
    ),
    :dot_com_bust_2000 => StressScenario(
        "Dot-Com Bust 2000-2002",
        "Tech bubble burst - NASDAQ fell ~78%",
        Dict(:equity => -0.45, :tech => -0.75, :bond => 0.15, :commodity => -0.10, :reit => -0.20),
        730
    ),
    :black_monday_1987 => StressScenario(
        "Black Monday 1987",
        "Single-day crash of 22.6%",
        Dict(:equity => -0.226, :bond => 0.02, :gold => 0.03),
        1
    ),
    :rate_shock_2022 => StressScenario(
        "2022 Rate Shock",
        "Fed aggressive tightening - bonds and stocks fell together",
        Dict(:equity => -0.25, :bond => -0.15, :reit => -0.30, :tech => -0.35),
        365
    ),
    :stagflation_1970s => StressScenario(
        "1970s Stagflation",
        "High inflation with economic stagnation",
        Dict(:equity => -0.30, :bond => -0.20, :commodity => 0.40, :gold => 0.50, :reit => -0.15),
        1095
    )
)

# ============================================================================
# Scenario Application Functions
# ============================================================================

"""
    apply_scenario(scenario, state, asset_classes)

Apply a stress scenario to a simulation state, returning a new stressed state.

# Arguments
- `scenario::StressScenario` - The stress scenario to apply
- `state::SimulationState{T}` - Current portfolio state
- `asset_classes::Dict{Symbol,Symbol}` - Mapping of asset symbols to asset classes

# Returns
- `SimulationState{T}` - New state with stressed prices

# Example
```julia
state = SimulationState(
    timestamp=DateTime(2024, 1, 1),
    cash=50_000.0,
    positions=Dict(:SPY => 100.0, :TLT => 50.0),
    prices=Dict(:SPY => 450.0, :TLT => 100.0)
)
asset_classes = Dict(:SPY => :equity, :TLT => :bond)
scenario = CRISIS_SCENARIOS[:financial_crisis_2008]
stressed_state = apply_scenario(scenario, state, asset_classes)
```
"""
function apply_scenario(
    scenario::StressScenario,
    state::SimulationState{T},
    asset_classes::Dict{Symbol,Symbol}
) where T

    new_prices = Dict{Symbol,T}()

    for (sym, price) in state.prices
        asset_class = get(asset_classes, sym, :equity)  # Default to equity
        shock = get(scenario.shocks, asset_class, 0.0)
        new_prices[sym] = price * (1 + shock)
    end

    SimulationState{T}(
        state.timestamp,
        state.cash,
        copy(state.positions),
        new_prices,
        copy(state.metadata)
    )
end

"""
    scenario_impact(scenario, state, asset_classes)

Compute the impact of a stress scenario on a portfolio.

# Arguments
- `scenario::StressScenario` - The stress scenario to evaluate
- `state::SimulationState` - Current portfolio state
- `asset_classes::Dict{Symbol,Symbol}` - Mapping of asset symbols to asset classes

# Returns
- `ScenarioImpact` - Detailed impact analysis including P&L and per-asset breakdown

# Example
```julia
state = SimulationState(
    timestamp=DateTime(2024, 1, 1),
    cash=50_000.0,
    positions=Dict(:SPY => 100.0, :TLT => 50.0),
    prices=Dict(:SPY => 450.0, :TLT => 100.0)
)
asset_classes = Dict(:SPY => :equity, :TLT => :bond)
scenario = CRISIS_SCENARIOS[:financial_crisis_2008]
impact = scenario_impact(scenario, state, asset_classes)
# impact.pnl < 0 (loss during crisis)
```
"""
function scenario_impact(
    scenario::StressScenario,
    state::SimulationState,
    asset_classes::Dict{Symbol,Symbol}
)
    initial_value = portfolio_value(state)
    stressed_state = apply_scenario(scenario, state, asset_classes)
    stressed_value = portfolio_value(stressed_state)

    pnl = stressed_value - initial_value
    pct_change = pnl / initial_value

    # Per-asset impacts
    asset_impacts = Dict{Symbol,Float64}()
    for (sym, qty) in state.positions
        old_val = qty * state.prices[sym]
        new_val = qty * stressed_state.prices[sym]
        asset_impacts[sym] = new_val - old_val
    end

    ScenarioImpact(
        scenario.name,
        initial_value,
        stressed_value,
        pnl,
        pct_change,
        asset_impacts
    )
end

# ============================================================================
# Exports
# ============================================================================

export StressScenario, ScenarioImpact, CRISIS_SCENARIOS
export apply_scenario, scenario_impact

end
