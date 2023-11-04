# By default, Julia/LLVM does not use fused multiply-add operations (FMAs).
# Since these FMAs can increase the performance of many numerical algorithms,
# we need to opt-in explicitly.
# See https://ranocha.de/blog/Optimizing_EC_Trixi for further details.
@muladd begin
#! format: noindent

"""
    AMMCallback(semi, controller [,adaptor=AdaptorAMM(semi)];
                interval,
                adapt_initial_condition=5)

Performs adaptive mesh movement (AMM) every `interval` time steps
for a given semidiscretization `semi` using the chosen `controller`.
"""
struct AMMCallback{Controller, Adaptor, Cache}
    controller::Controller
    interval::Int
    adapt_initial_condition::Int
    adaptor::Adaptor
    amm_cache::Cache
end

function AMMCallback(semi, controller, adaptor;
                     interval,
                     adapt_initial_condition = 5)
    # check arguments
    if !(interval isa Integer && interval >= 0)
        throw(ArgumentError("`interval` must be a non-negative integer (provided `interval = $interval`)"))
    end

    # AMM every `interval` time steps, but not after the final step
    # With error-based step size control, some steps can be rejected. Thus,
    #   `integrator.iter >= integrator.stats.naccept`
    #    (total #steps)       (#accepted steps)
    # We need to check the number of accepted steps since callbacks are not
    # activated after a rejected step.
    if interval > 0
        condition = (u, t, integrator) -> ((integrator.stats.naccept % interval == 0) &&
                                           !(integrator.stats.naccept == 0 &&
                                             integrator.iter > 0) &&
                                           !isfinished(integrator))
    else # disable the AMM callback except possibly for initial adaptation during initialization
        condition = (u, t, integrator) -> false
    end

    to_refine = Int[]
    to_coarsen = Int[]
    amm_cache = (; to_refine, to_coarsen)

    amm_callback = AMMCallback{typeof(controller), typeof(adaptor), typeof(amm_cache)}(controller,
                                                                                       interval,
                                                                                       adapt_initial_condition,
                                                                                       adaptor,
                                                                                       amm_cache)

    DiscreteCallback(condition, amm_callback,
                     save_positions = (false, false),
                     initialize = initialize!)
end

function AMMCallback(semi, controller; kwargs...)
    adaptor = AdaptorAMM(semi)
    AMMCallback(semi, controller, adaptor; kwargs...)
end

function AdaptorAMM(semi; kwargs...)
    mesh, _, solver, _ = mesh_equations_solver_cache(semi)
    AdaptorAMM(mesh, solver; kwargs...)
end

function AdaptorAMM(mesh, solver; kwargs...)
  println("Do nothing now for AdaptorAMM")
end

# TODO: Taal bikeshedding, implement a method with less information and the signature
# function Base.show(io::IO, cb::DiscreteCallback{<:Any, <:AMMCallback})
#   @nospecialize cb # reduce precompilation time
#
#   amm_callback = cb.affect!
#   print(io, "AMMCallback")
# end
function Base.show(io::IO, mime::MIME"text/plain",
                   cb::DiscreteCallback{<:Any, <:AMMCallback})
    @nospecialize cb # reduce precompilation time

    if get(io, :compact, false)
        show(io, cb)
    else
        amm_callback = cb.affect!

        summary_header(io, "AMMCallback")
        summary_line(io, "controller", amm_callback.controller |> typeof |> nameof)
        show(increment_indent(io), mime, amm_callback.controller)
        summary_line(io, "interval", amm_callback.interval)
        summary_line(io, "adapt IC",
                     amm_callback.adapt_initial_condition > 0 ? "yes" : "no")
        if amm_callback.adapt_initial_condition > 0
            summary_line(io, "│ initial adaptation",
                         amm_callback.adapt_initial_condition)
        end
        summary_footer(io)
    end
end

# The function below is used to control the output depending on whether or not AMM is enabled.
"""
    uses_amm(callback)

Checks whether the provided callback or `CallbackSet` is an [`AMMCallback`](@ref)
or contains one.
"""
uses_amm(cb) = false
function uses_amm(cb::DiscreteCallback{Condition, Affect!}) where {Condition,
                                                                   Affect! <:
                                                                   AMMCallback}
    true
end
uses_amm(callbacks::CallbackSet) = mapreduce(uses_amm, |, callbacks.discrete_callbacks)

# function get_element_variables!(element_variables, u, mesh, equations, solver, cache,
                                # amm_callback::AMMCallback; kwargs...)
    # get_element_variables!(element_variables, u, mesh, equations, solver, cache,
                           # amm_callback.controller, amm_callback; kwargs...)
# end

function initialize!(cb::DiscreteCallback{Condition, Affect!}, u, t,
                     integrator) where {Condition, Affect! <: AMMCallback}
    amm_callback = cb.affect!
    semi = integrator.p

    @trixi_timeit timer() "initial condition AMM" if amm_callback.adapt_initial_condition > 0
        # iterate until mesh does not change anymore
        for i = 1:amm_callback.adapt_initial_condition
            println("initialize, i, iter, t: ", i, ", ", integrator.iter, ", ", integrator.t)
            amm_callback(integrator)
            println("finish op init")
            compute_coefficients!(integrator.u, t, semi)
            u_modified!(integrator, true)
        end
    end

    return nothing
end

function (amm_callback::AMMCallback)(integrator; kwargs...)
    u_ode = integrator.u
    semi = integrator.p

    @trixi_timeit timer() "AMM" begin
        println("do something 1, t: ", integrator.iter, ", ", integrator.t)
        amm_callback(u_ode, semi, integrator.t, integrator.iter; kwargs...)
        # resize!(integrator, length(u_ode))
        # u_modified!(integrator, true)
    end
end

@inline function (amm_callback::AMMCallback)(u_ode::AbstractVector,
                                             semi::SemidiscretizationHyperbolic,
                                             t, iter;
                                             kwargs...)
    # Note that we don't `wrap_array` the vector `u_ode` to be able to `resize!`
    # it when doing AMM while still dispatching on the `mesh` etc.
    println("do something 2")
    amm_callback(u_ode, mesh_equations_solver_cache(semi)..., semi, t, iter; kwargs...)
end

include("amm_callbacks_step_2d.jl")

end # @muladd
