# Generate visualization screenshots for documentation
# Run: julia --project=. docs/generate_screenshots.jl

using Pkg
Pkg.activate(joinpath(@__DIR__, ".."))

using QuantNova
using CairoMakie
using Dates
using Random
using Statistics: mean, std
using LinearAlgebra: I, diag

CairoMakie.activate!()

# Use CairoMakie's save to avoid conflict with QuantNova.save
const savefig = CairoMakie.save

const ASSETS_DIR = joinpath(@__DIR__, "src", "assets")
mkpath(ASSETS_DIR)

println("Generating visualization screenshots...")
println("Output directory: $ASSETS_DIR")
println()

# =============================================================================
# Generate Sample Data
# =============================================================================

Random.seed!(2024)

# 2 years of daily data
n_days = 252 * 2
timestamps = [DateTime(2023, 1, 1) + Day(i) for i in 1:n_days]

# Simple but realistic return series: drift + noise
# ~10% annual return with ~15% annual vol
daily_returns = 0.0004 .+ 0.0095 .* randn(n_days)

# Compute equity curve
equity = zeros(n_days)
equity[1] = 10000.0
for i in 2:n_days
    equity[i] = equity[i-1] * (1 + daily_returns[i])
end

# Compute metrics
sharpe = mean(daily_returns) / std(daily_returns) * sqrt(252)
max_dd = maximum(accumulate(max, equity) .- equity) / maximum(equity)
cagr = (equity[end] / equity[1])^(252/n_days) - 1
vol_ann = std(daily_returns) * sqrt(252)

# Create BacktestResult
result = BacktestResult(
    10000.0,
    equity[end],
    equity,
    daily_returns[2:end],
    timestamps,
    Fill[],
    [Dict{Symbol,Float64}() for _ in 1:n_days],
    Dict{Symbol,Float64}(
        :sharpe_ratio => sharpe,
        :max_drawdown => -max_dd,
        :annualized_return => cagr,
        :volatility => vol_ann,
        :total_return => (equity[end] - equity[1]) / equity[1]
    )
)

println("Generated sample BacktestResult:")
println("  Period: $(Date(timestamps[1])) to $(Date(timestamps[end]))")
println("  Final value: \$$(round(equity[end], digits=2))")
println("  Sharpe: $(round(sharpe, digits=2))")
println("  Total return: $(round((equity[end]/equity[1] - 1)*100, digits=1))%")
println()

# =============================================================================
# Helper: Generate for both themes
# =============================================================================

function generate_both_themes(name, view_or_fn; result=result, size=(900, 500), kwargs...)
    for theme in [:light, :dark]
        println("  → $name ($theme)...")

        if view_or_fn isa Symbol
            spec = visualize(result, view_or_fn; theme=theme, kwargs...)
        else
            spec = view_or_fn(theme)
        end

        fig = render(spec)
        resize!(fig, size...)

        filename = "viz-$name-$theme.png"
        savefig(joinpath(ASSETS_DIR, filename), fig; px_per_unit=2)
        println("    ✓ $filename")
    end
end

# =============================================================================
# Backtest Visualizations
# =============================================================================

println("=" ^ 50)
println("Backtest Visualizations")
println("=" ^ 50)

# 1. Equity Curve
println("\n1. Equity Curve")
generate_both_themes("equity", :equity; title="Portfolio Performance", size=(900, 500))

# 2. Drawdown Chart
println("\n2. Drawdown Chart")
generate_both_themes("drawdown", :drawdown; title="Underwater Equity", size=(900, 400))

# 3. Returns Distribution
println("\n3. Returns Distribution")
generate_both_themes("returns", :returns; title="Daily Returns", size=(800, 500))

# 4. Rolling Metrics
println("\n4. Rolling Metrics")
generate_both_themes("rolling", :rolling; title="Rolling 63-Day Metrics", window=63, size=(900, 600))

# 5. Dashboard (composite)
println("\n5. Dashboard")
generate_both_themes("dashboard", :dashboard; size=(1200, 800))

# =============================================================================
# Portfolio Visualizations
# =============================================================================

println()
println("=" ^ 50)
println("Portfolio Visualizations")
println("=" ^ 50)

# Create sample optimization data
n_assets = 5
asset_names = [:AAPL, :MSFT, :GOOGL, :AMZN, :META]

# Sample covariance matrix
Random.seed!(42)
A = randn(n_assets, n_assets) * 0.2
Σ = A' * A + 0.1 * I
Σ = (Σ + Σ') / 2

# Expected returns (annualized)
μ = [0.12, 0.10, 0.15, 0.08, 0.11]

# Run optimization
opt_result = optimize(MinimumVariance(Σ))
println("\nMin-Variance weights: $(round.(opt_result.weights, digits=3))")

# 6. Portfolio Weights
println("\n6. Portfolio Weights")
generate_both_themes("weights", theme -> visualize(opt_result, :weights; theme=theme, title="Portfolio Allocation", assets=asset_names); size=(800, 500))

# 7. Efficient Frontier
println("\n7. Efficient Frontier")
generate_both_themes("frontier", theme -> visualize(opt_result, :frontier; theme=theme, title="Risk-Return Tradeoff", μ=μ, Σ=Σ); size=(800, 600))

# =============================================================================
# Summary
# =============================================================================

println()
println("=" ^ 50)
println("Screenshot generation complete!")
println("=" ^ 50)
println()

files = readdir(ASSETS_DIR)
png_files = filter(f -> endswith(f, ".png"), files)
println("Generated $(length(png_files)) screenshots:")

let total_size = 0.0
    for f in sort(png_files)
        size_kb = filesize(joinpath(ASSETS_DIR, f)) / 1024
        total_size += size_kb
        println("  • $f ($(round(size_kb, digits=1)) KB)")
    end
    println()
    println("Total size: $(round(total_size / 1024, digits=2)) MB")
end
