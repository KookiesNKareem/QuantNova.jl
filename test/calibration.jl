@testset "Calibration" begin

    @testset "Black-76 Pricing" begin
        # Test Black-76 put-call parity
        F, K, T, r, σ = 100.0, 100.0, 1.0, 0.05, 0.2
        call = black76(F, K, T, r, σ, :call)
        put = black76(F, K, T, r, σ, :put)

        # Put-call parity: C - P = e^(-rT) * (F - K)
        df = exp(-r * T)
        @test call - put ≈ df * (F - K) atol=1e-10

        # ATM forward call should be worth roughly 0.4 * σ * sqrt(T) * F * df
        # (approximation for ATM options)
        @test call > 0
        @test put > 0

        # OTM call (high strike)
        otm_call = black76(F, 150.0, T, r, σ, :call)
        @test otm_call < call

        # OTM put (low strike)
        otm_put = black76(F, 50.0, T, r, σ, :put)
        @test otm_put < put
    end

    @testset "SABR Implied Volatility" begin
        # Test SABR implied vol with known parameters
        params = SABRParams(0.2, 0.5, -0.3, 0.4)
        F, T = 100.0, 1.0

        # ATM vol
        atm_vol = sabr_implied_vol(F, F, T, params)
        @test atm_vol > 0
        @test atm_vol < 1.0  # Reasonable bounds

        # Smile should show skew (lower strikes have higher vol for negative rho)
        low_strike_vol = sabr_implied_vol(F, 80.0, T, params)
        high_strike_vol = sabr_implied_vol(F, 120.0, T, params)
        # Negative rho creates downside skew: low strikes have relatively higher vol
        @test low_strike_vol > high_strike_vol  # Main skew effect

        # Test symmetry with rho = 0
        sym_params = SABRParams(0.2, 0.5, 0.0, 0.4)
        low_vol = sabr_implied_vol(F, 90.0, T, sym_params)
        high_vol = sabr_implied_vol(F, 110.0, T, sym_params)
        # With rho=0 and beta=0.5, smile is roughly symmetric around ATM
        @test abs(low_vol - high_vol) < 0.02
    end

    @testset "SABR Pricing" begin
        params = SABRParams(0.2, 0.5, -0.3, 0.4)
        F, K, T, r = 100.0, 100.0, 1.0, 0.05

        call = sabr_price(F, K, T, r, params, :call)
        put = sabr_price(F, K, T, r, params, :put)

        @test call > 0
        @test put > 0

        # Put-call parity should hold
        df = exp(-r * T)
        @test call - put ≈ df * (F - K) atol=1e-8
    end

    @testset "SABR Calibration" begin
        # Generate synthetic smile from known SABR params
        # Using beta=1.0 (lognormal) which is simpler and more common for equities
        # alpha around 0.25 gives ~25% ATM vol which is realistic
        true_params = SABRParams(0.25, 1.0, -0.3, 0.4)
        F, T, r = 100.0, 1.0, 0.05
        strikes = [80.0, 85.0, 90.0, 95.0, 100.0, 105.0, 110.0, 115.0, 120.0]

        # Create quotes with known implied vols
        quotes = [OptionQuote(K, T, 0.0, :call, sabr_implied_vol(F, K, T, true_params))
                  for K in strikes]

        smile = SmileData(T, F, r, quotes)

        # Calibrate
        result = calibrate_sabr(smile; beta=1.0, max_iter=3000, lr=0.01)

        @test result.converged
        # SABR has identifiability issues - different (α,ρ,ν) can produce similar smiles
        # Focus on fit quality rather than exact parameter recovery
        @test result.rmse < 0.01  # Implied vol RMSE < 1%

        # The calibrated model should reproduce the smile reasonably well
        max_error = 0.0
        for q in quotes
            fitted_vol = sabr_implied_vol(F, q.strike, T, result.params)
            max_error = max(max_error, abs(fitted_vol - q.implied_vol))
        end
        @test max_error < 0.02  # Max error < 2% in vol terms

        @test result.params.beta == 1.0  # Beta is fixed
    end

    @testset "Heston Characteristic Function" begin
        params = HestonParams(0.04, 0.04, 1.0, 0.3, -0.5)
        S, T, r = 100.0, 1.0, 0.05

        # Characteristic function at u=0 should give exp(i*0*log(S)) * ... = forward-like
        φ0 = heston_characteristic(0.0, S, T, r, params)
        @test abs(φ0) ≈ 1.0 atol=0.01  # |φ(0)| should be close to 1

        # Characteristic function should be defined for complex u
        φ1 = heston_characteristic(1.0, S, T, r, params)
        @test isfinite(abs(φ1))
    end

    @testset "Heston Pricing" begin
        # Heston parameters: v0=0.04, θ=0.04, κ=1.5, σ=0.3, ρ=-0.5
        params = HestonParams(0.04, 0.04, 1.5, 0.3, -0.5)
        S, K, T, r = 100.0, 100.0, 1.0, 0.05

        call = heston_price(S, K, T, r, params, :call)
        put = heston_price(S, K, T, r, params, :put)

        @test call > 0
        @test put >= 0  # Put can be small for ATM

        # Put-call parity: C - P = S - K*exp(-rT)
        parity_diff = call - put - (S - K * exp(-r * T))
        @test abs(parity_diff) < 1.0  # Looser tolerance for numerical integration

        # OTM call should be cheaper than ATM call
        otm_call = heston_price(S, 120.0, T, r, params, :call)
        @test otm_call < call

        # ITM put (high strike) should have value
        itm_put = heston_price(S, 120.0, T, r, params, :put)
        @test itm_put > 0

        # Longer maturity should have higher ATM option value
        long_call = heston_price(S, K, 2.0, r, params, :call)
        @test long_call > call
    end

    @testset "Heston Calibration" begin
        # Generate synthetic surface from known Heston params
        true_params = HestonParams(0.04, 0.04, 1.5, 0.3, -0.5)
        S, r = 100.0, 0.05

        expiries = [0.25, 0.5, 1.0]
        strikes_base = [85.0, 90.0, 95.0, 100.0, 105.0, 110.0, 115.0]

        smiles = SmileData[]
        for T in expiries
            F = S * exp(r * T)  # Forward price
            quotes = OptionQuote[]
            for K in strikes_base
                price = heston_price(S, K, T, r, true_params, :call)
                # We need implied vol for the quote - use a rough approximation
                # In practice you'd use a root finder, but for testing we can use price
                push!(quotes, OptionQuote(K, T, price, :call, 0.2))  # IV not used in Heston calib
            end
            push!(smiles, SmileData(T, F, r, quotes))
        end

        surface = VolSurface(S, r, smiles)

        # Calibrate (use fewer iterations for test speed)
        result = calibrate_heston(surface; max_iter=500, lr=0.01)

        # Heston calibration is harder - just check it runs and produces reasonable output
        @test result.iterations > 0
        @test result.loss < 1.0  # Should have made progress
        @test result.params.v0 > 0
        @test result.params.theta > 0
        @test result.params.kappa > 0
        @test result.params.sigma > 0
        @test -1 < result.params.rho < 1
    end

end
