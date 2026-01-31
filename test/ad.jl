using Test
using Quasar

@testset "AD Backend System" begin
    @testset "Backend types exist" begin
        @test PureJuliaBackend <: AbstractADBackend
        @test ForwardDiffBackend <: AbstractADBackend
        @test ReactantBackend <: AbstractADBackend
    end

    @testset "Backend selection" begin
        # Default backend should be ForwardDiff (most stable)
        @test current_backend() isa ForwardDiffBackend

        # Can change backend
        set_backend!(PureJuliaBackend())
        @test current_backend() isa PureJuliaBackend

        # Reset for other tests
        set_backend!(ForwardDiffBackend())
    end

    @testset "Gradient computation" begin
        f(x) = sum(x.^2)
        x = [1.0, 2.0, 3.0]

        # ForwardDiff backend
        set_backend!(ForwardDiffBackend())
        g = gradient(f, x)
        @test g ≈ [2.0, 4.0, 6.0]

        # PureJulia backend (finite differences)
        set_backend!(PureJuliaBackend())
        g_fd = gradient(f, x)
        @test g_fd ≈ [2.0, 4.0, 6.0] atol=1e-6
    end
end
