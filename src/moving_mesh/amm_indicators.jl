# By default, Julia/LLVM does not use fused multiply-add operations (FMAs).
# Since these FMAs can increase the performance of many numerical algorithms,
# we need to opt-in explicitly.
# See https://ranocha.de/blog/Optimizing_EC_Trixi for further details.
@muladd begin
#! format: noindent

abstract type AbstractIndicator end

function create_cache(typ::Type{IndicatorType},
                      semi) where {IndicatorType <: AbstractIndicator}
    create_cache(typ, mesh_equations_solver_cache(semi)...)
end

function get_element_variables!(element_variables, indicator::AbstractIndicator,
                                ::VolumeIntegralShockCapturingHG)
    element_variables[:indicator_shock_capturing] = indicator.cache.alpha
    return nothing
end

"""
    IndicatorGradient(semi::AbstractSemidiscretization;
                              alpha_max=0.5,
                              alpha_min=0.001,
                              alpha_scale=10.0,
                              alpha_smooth=5,
                              variable)

Indicator used for shock-capturing (when passing the `equations` and the `basis`)
or adaptive mesh movement (AMM, when passing the `semi`).

See also [`VolumeIntegralShockCapturingHG`](@ref).

## References

- Hennemann, Gassner (2020)
  "A provably entropy stable subcell shock capturing approach for high order split form DG"
  [arXiv: 2008.12044](https://arxiv.org/abs/2008.12044)
"""
struct IndicatorGradient{RealT <: Real, Variable, Cache} <: AbstractIndicator
    alpha_max::RealT
    alpha_min::RealT
    alpha_scale::RealT
    alpha_smooth::Int
    variable::Variable
    cache::Cache
end

# this method is used when the indicator is constructed as for AMM
function IndicatorGradient(semi::AbstractSemidiscretization;
                                   alpha_max = 0.5,
                                   alpha_min = 0.001,
                                   alpha_scale = 10.0,
                                   alpha_smooth = 5,
                                   variable)
    alpha_max, alpha_min = promote(alpha_max, alpha_min)
    cache = create_cache(IndicatorGradient, semi)
    IndicatorGradient{typeof(alpha_max), typeof(variable), typeof(cache)}(alpha_max,
                                                                          alpha_min,
                                                                          alpha_scale,
                                                                          alpha_smooth,
                                                                          variable,
                                                                          cache)
end

function Base.show(io::IO, indicator::IndicatorGradient)
    @nospecialize indicator # reduce precompilation time

    print(io, "IndicatorGradient(")
    print(io, indicator.variable)
    print(io, ", alpha_max=", indicator.alpha_max)
    print(io, ", alpha_min=", indicator.alpha_min)
    print(io, ", alpha_scale=", indicator.alpha_scale)
    print(io, ", alpha_smooth=", indicator.alpha_smooth)
    print(io, ")")
end

function Base.show(io::IO, ::MIME"text/plain", indicator::IndicatorGradient)
    @nospecialize indicator # reduce precompilation time
    setup = [
        "indicator variable" => indicator.variable,
        "max. α" => indicator.alpha_max,
        "min. α" => indicator.alpha_min,
        "scale. α" => indicator.alpha_scale,
        "smooth. α" => indicator.alpha_smooth,
    ]
    summary_box(io, "IndicatorGradient", setup)
end

function (indicator_gradient::IndicatorGradient)(u, mesh, equations, dg::DGSEM, cache;
                                                   kwargs...)
    @unpack alpha_scale, alpha_smooth = indicator_gradient
    @unpack alpha, alpha_tmp = indicator_gradient.cache
    # TODO: Taal refactor, when to `resize!` stuff changed possibly by AMM?
    #       Shall we implement `resize!(semi::AbstractSemidiscretization, new_size)`
    #       or just `resize!` whenever we call the relevant methods as we do now?
    resize!(alpha, nelements(dg, cache))
    if alpha_smooth > 0
        resize!(alpha_tmp, nelements(dg, cache))
    end

    max_gradient = Threads.Atomic{eltype(alpha)}(0.0)
    min_gradient = Threads.Atomic{eltype(alpha)}(1.0e10)

    @threaded for element in eachelement(dg, cache)
        # This is dispatched by mesh dimension.
        # Use this function barrier and unpack inside to avoid passing closures to
        # Polyester.jl with `@batch` (`@threaded`).
        # Otherwise, `@threaded` does not work here with Julia ARM on macOS.
        # See https://github.com/JuliaSIMD/Polyester.jl/issues/88.
        gradient = calc_indicator_gradient!(indicator_gradient, u,
                                 element, mesh, equations, dg, cache)
        Threads.atomic_max!(max_gradient, gradient)
        Threads.atomic_min!(min_gradient, gradient)
    end

    println("max: ", max_gradient.value)
    println("min: ", min_gradient.value)
    @threaded for element in eachelement(dg, cache)
      alpha[element] = sqrt(1 + alpha_scale*alpha[element]/max_gradient.value)
    end

    for i = 1:alpha_smooth
        apply_smoothing!(mesh, alpha, alpha_tmp, dg, cache)
    end

    return alpha
end

include("amm_indicators_2d.jl")

end # @muladd
