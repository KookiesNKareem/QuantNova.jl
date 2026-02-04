# Validation

This page documents how QuantNova is validated for numerical correctness. It focuses on **how to reproduce** checks rather than reporting static numbers.

## What We Validate

- **Black-Scholes parity and edge cases** (e.g., `T → 0`, `σ → 0`, deep ITM/OTM).
- **Greeks accuracy** versus analytical formulas.
- **Monte Carlo sanity checks** versus analytical prices where available.
- **Yield curve bootstrapping and bond analytics** consistency.
- **SABR / Heston calibration** convergence and stability on known inputs.
- **Statistical metrics** (Sharpe, confidence intervals) against reference formulas.

## How to Reproduce

### 1) Core Test Suite

Runs the unit tests that cover correctness and edge cases:

```bash
julia --project=. -e 'using Pkg; Pkg.test()'
```

Key test files (see `test/`):
- `test/instruments.jl`
- `test/ad.jl`
- `test/interest_rates.jl`
- `test/montecarlo.jl`
- `test/statistics.jl`
- `test/calibration.jl`

### 2) Accuracy Benchmarks

We keep targeted accuracy scripts under:

- `benchmarks/accuracy/black_scholes_parity.jl`
- `benchmarks/accuracy/heston_reference.jl`

Run a specific script:

```bash
julia --project=. benchmarks/accuracy/black_scholes_parity.jl
```

### 3) QuantLib Cross-Checks (Optional)

If you have QuantLib installed, you can compare outputs directly using:

```bash
julia --project=. benchmarks/comparison/quantlib_comparison.jl
```

This script validates prices and Greeks and reports differences.

## Notes on Units and Conventions

- **Theta:** QuantNova returns theta per year. Some libraries return per day.
- **Vega/Rho:** QuantNova reports per 1% change (scaled by 0.01), which matches most practitioner conventions.

When comparing results, always align conventions first.

## Interpreting Results

Validation focuses on:

- **Exact identities** (e.g., put-call parity).
- **Small absolute error** for analytical comparisons.
- **Consistency across implementations** (e.g., AD vs analytical).

If any check fails, open the relevant test file and review assumptions (inputs, conventions, or units).
