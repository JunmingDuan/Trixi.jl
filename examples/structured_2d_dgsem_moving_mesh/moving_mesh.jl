# By default, Julia/LLVM does not use fused multiply-add operations (FMAs).
# Since these FMAs can increase the performance of many numerical algorithms,
# we need to opt-in explicitly.
# See https://ranocha.de/blog/Optimizing_EC_Trixi for further details.
using MuladdMacro
@muladd begin
    #! format: noindent
    
"""
    MMCallback(semi, controller [,adaptor=AdaptorMM(semi)];
               interval,
               adapt_initial_condition=5)

Performs adaptive mesh refinement (AMR) every `interval` time steps
for a given semidiscretization `semi` using the chosen `controller`.
"""
struct MMCallback{Controller, Adaptor, Cache}
    controller::Controller
    interval::Int
    adapt_initial_condition::Int
    adaptor::Adaptor
    mm_cache::Cache
end

function MMCallback(semi, controller, adaptor;
    interval = 1,
    adapt_initial_condition = 5)
# check arguments
if !(interval isa Integer && interval >= 0)
throw(ArgumentError("`interval` must be a non-negative integer (provided `interval = $interval`)"))
end

# MM every `interval` time steps, but not after the final step
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
else # disable the MM callback except possibly for initial adaptation during initialization
condition = (u, t, integrator) -> false
end

mm_cache = (;)

mm_callback = MMCallback{typeof(controller), typeof(adaptor), typeof(mm_cache)}(controller,
                                                      interval,
                                                      adapt_initial_condition,
                                                      adaptor,
                                                      mm_cache)

    DiscreteCallback(condition, mm_callback,
        save_positions = (false, false),
        initialize = initialize!)
end

function MMCallback(semi, controller; kwargs...)
    adaptor = AdaptorMM(semi)
    MMCallback(semi, controller, adaptor; kwargs...)
end

function AdaptorMM(semi; kwargs...)
    mesh, _, solver, _ = mesh_equations_solver_cache(semi)
    AdaptorMM(mesh, solver; kwargs...)
end

function Base.show(io::IO, mime::MIME"text/plain",
    cb::DiscreteCallback{<:Any, <:MMCallback})
    @nospecialize cb # reduce precompilation time

    if get(io, :compact, false)
        show(io, cb)
    else
        mm_callback = cb.affect!

        summary_header(io, "MMCallback")
        summary_line(io, "controller", mm_callback.controller |> typeof |> nameof)
        show(increment_indent(io), mime, mm_callback.controller)
        summary_line(io, "interval", mm_callback.interval)
        summary_line(io, "adapt IC",
            mm_callback.adapt_initial_condition ? "yes" : "no")
        if mm_callback.adapt_initial_condition
            summary_line(io, "│ only refine",
                    mm_callback.adapt_initial_condition_only_refine ? "yes" :
                    "no")
        end
        summary_footer(io)
    end
end

uses_mm(cb) = false
function uses_mm(cb::DiscreteCallback{Condition, Affect!}) where {Condition,
                                                                   Affect! <:
                                                                   MMCallback}
    true
end
uses_mm(callbacks::CallbackSet) = mapreduce(uses_mm, |, callbacks.discrete_callbacks)

function initialize!(cb::DiscreteCallback{Condition, Affect!}, u, t,
    integrator) where {Condition, Affect! <: MMCallback}
    mm_callback = cb.affect!
    semi = integrator.p

    @trixi_timeit timer() "initial condition MM" if mm_callback.adapt_initial_condition
        # iterate until mesh does not change anymore
        has_changed = mm_callback(integrator,
                        only_refine = mm_callback.adapt_initial_condition_only_refine)
        while has_changed
            compute_coefficients!(integrator.u, t, semi)
            u_modified!(integrator, true)
            has_changed = mm_callback(integrator,
                                only_refine = mm_callback.adapt_initial_condition_only_refine)
        end
    end

    return nothing
end

function (mm_callback::MMCallback)(integrator; kwargs...)
    u_ode = integrator.u
    semi = integrator.p

    @trixi_timeit timer() "MM" begin
        has_changed = mm_callback(u_ode, semi,
                                   integrator.t, integrator.iter; kwargs...)
        if has_changed
            resize!(integrator, length(u_ode))
            u_modified!(integrator, true)
        end
    end

    return has_changed
end

@inline function (mm_callback::MMCallback)(u_ode::AbstractVector,
    semi::SemidiscretizationHyperbolic,
    t, iter;
    kwargs...)
    # Note that we don't `wrap_array` the vector `u_ode` to be able to `resize!`
    # it when doing MM while still dispatching on the `mesh` etc.
    mm_callback(u_ode, mesh_equations_solver_cache(semi)..., semi, t, iter; kwargs...)
end

end # @muladd
