# Nova.jl - Differentiable Quantitative Finance Library

I'm sharing **Nova.jl**, a quant finance library built around two ideas: a unified API and automatic differentiation everywhere.

**GitHub:** https://github.com/KookiesNKareem/Nova.jl
**Docs:** https://KookiesNKareem.github.io/Nova.jl/dev/

## Why Nova?

**One API, everything differentiable.** Price an option, compute Greeks, calibrate a model, optimize a portfolio - it's all the same interface, and gradients flow through automatically.

```julia
using Nova

# Price an option
option = EuropeanOption("AAPL", 155.0, 0.5, :call)
price(option, state)

# Greeks - just one function, AD handles it
greeks = compute_greeks(option, state)

# Calibrate SABR - same gradient machinery
result = calibrate_sabr(smile_data)

# Portfolio optimization - gradients work here too
result = optimize(MeanVariance(μ, Σ); target_return=0.10)
```

No separate "analytical Greeks" vs "numerical Greeks" APIs. No special calibration code. The AD system handles it.

**Backend-agnostic.** Write once, run on CPU or GPU:

```julia
# Default: ForwardDiff (CPU)
gradient(f, x)

# Switch to GPU
with_backend(ReactantBackend()) do
    gradient(f, x)  # Same code, now on GPU
end
```

## What's included

- **Pricing:** European, American, Asian, Barrier options; Monte Carlo with variance reduction
- **Models:** Black-Scholes, SABR, Heston
- **Optimization:** Mean-variance, risk parity, Black-Litterman
- **Risk:** VaR, CVaR, Sharpe, drawdown
- **Rates:** Curves, bonds, caps/floors, swaptions
- **Backtesting:** Strategy testing with execution models

## Installation

```julia
using Pkg
Pkg.add(url="https://github.com/KookiesNKareem/Nova.jl")
```

Feedback and contributions welcome.
