# By default, Julia/LLVM does not use fused multiply-add operations (FMAs).
# Since these FMAs can increase the performance of many numerical algorithms,
# we need to opt-in explicitly.
# See https://ranocha.de/blog/Optimizing_EC_Trixi for further details.
@muladd begin
#! format: noindent

# struct SimpleDuanTang end
# default_movingmesh_solver() = SimpleDuanTang()

"""
    SemidiscretizationMovingMesh

    A struct containing everything needed to describe a spatial semidiscretization
    of a hyperbolic conservation law on moving meshes.
"""

struct SemidiscretizationMovingMesh{Mesh, Equations,
                                    InitialCondition,
                                    BoundaryConditions,
                                    Solver,
                                    Cache} <: AbstractSemidiscretization
    mesh::Mesh
    equations::Equations

    initial_condition::InitialCondition
    boundary_conditions::BoundaryConditions
    solver::Solver
    cache::Cache
    performance_counter::PerformanceCounter

    function SemidiscretizationMovingMesh{Mesh, Equations,
                                          InitialCondition,
                                          BoundaryConditions,
                                          Solver,
                                          Cache
                                          }(mesh::Mesh, equations::Equations,
                                            initial_condition::InitialCondition,
                                            boundary_conditions::BoundaryConditions,
                                            solver::Solver,
                                            cache::Cache,
                                            ) where {Mesh,
                                                     Equations,
                                                     InitialCondition,
                                                     BoundaryConditions,
                                                     Solver,
                                                     Cache}
        @assert ndims(mesh) == ndims(equations)
        performance_counter = PerformanceCounter()

        new(mesh, equations, initial_condition,
            boundary_conditions,
            solver, cache, performance_counter)
    end
end

"""
    SemidiscretizationMovingMesh(mesh, equations, initial_condition, solver;
                                 solver_movingmesh=default_movingmesh_solver(),
                                 source_terms=nothing,
                                 boundary_conditions=boundary_condition_periodic,
                                 boundary_conditions_movingmesh=boundary_condition_periodic,
                                 RealT=real(solver),
                                 uEltype=RealT,
                                 initial_caches=(NamedTuple(), NamedTuple()))

Construct a semidiscretization of a hyperbolic PDE on moving meshes.
"""
function SemidiscretizationMovingMesh(mesh, equations::Advection, initial_condition, solver;
                                      boundary_conditions = boundary_condition_periodic,
                                      # `RealT` is used as real type for node locations etc.
                                      # while `uEltype` is used as element type of solutions etc.
                                      RealT = real(solver), uEltype = RealT,
                                      initial_cache = NamedTuple())

    cache = (; create_cache(mesh, equations, solver, RealT, uEltype)...,
            initial_cache...)                                    
    SemidiscretizationMovingMesh(mesh, equations, initial_condition, solver;
                                 solver_movingmesh,
                                 boundary_conditions, boundary_conditions_movingmesh,
                                 source_terms,
                                 initial_cache = initial_cache_hyperbolic,
                                 initial_cache_movingmesh = initial_cache_movingmesh)
end

function SemidiscretizationMovingMesh(mesh, equations, initial_condition, solver;
                                      solver_movingmesh = default_movingmesh_solver(),
                                      source_terms = nothing,
                                      boundary_conditions = boundary_condition_periodic,
                                      boundary_conditions_movingmesh = boundary_condition_periodic,
                                      # `RealT` is used as real type for node locations etc.
                                      # while `uEltype` is used as element type of solutions etc.
                                      RealT = real(solver), uEltype = RealT,
                                      initial_cache = NamedTuple(),
                                      initial_cache_movingmesh = NamedTuple())

    cache = (; create_cache(mesh, equations, solver, RealT, uEltype)...,
             initial_cache...)
    _boundary_conditions = digest_boundary_conditions(boundary_conditions, mesh, solver,
                                                      cache)
    _boundary_conditions_movingmesh = digest_boundary_conditions(boundary_conditions_movingmesh, mesh, solver,
                                                      cache)
    cache_movingmesh = (; create_cache_movingmesh(mesh, solver, RealT, uEltype)...,
                        initial_cache_movingmesh...)

    SemidiscretizationMovingMesh{typeof(mesh), typeof(equations),
                                           typeof(initial_condition),
                                           typeof(_boundary_conditions),
                                           typeof(_boundary_conditions_movingmesh),
                                           typeof(source_terms),
                                           typeof(solver),
                                           typeof(solver_movingmesh),
                                           typeof(cache),
                                           typeof(cache_movingmesh)}(mesh, equations,
                                                                      initial_condition,
                                                                      _boundary_conditions,
                                                                      _boundary_conditions_movingmesh,
                                                                      source_terms,
                                                                      solver,
                                                                      solver_movingmesh,
                                                                      cache,
                                                                      cache_movingmesh)
  end

# Create a new semidiscretization but change some parameters compared to the input.
# `Base.similar` follows a related concept but would require us to `copy` the `mesh`,
# which would impact the performance. Instead, `SciMLBase.remake` has exactly the
# semantics we want to use here. In particular, it allows us to re-use mutable parts,
# e.g. `remake(semi).mesh === semi.mesh`.
function remake(semi::SemidiscretizationMovingMesh;
                uEltype = real(semi.solver),
                mesh = semi.mesh,
                equations = semi.equations,
                initial_condition = semi.initial_condition,
                solver = semi.solver,
                solver_movingmesh = semi.solver_movingmesh,
                source_terms = semi.source_terms,
                boundary_conditions = semi.boundary_conditions,
                boundary_conditions_movingmesh = semi.boundary_conditions_movingmesh)
    # TODO: Which parts do we want to `remake`? At least the solver needs some
    #       special care if shock-capturing volume integrals are used (because of
    #       the indicators and their own caches...).
    SemidiscretizationMovingMesh(mesh, equations, initial_condition, solver;
                                           solver_movingmesh, source_terms,
                                           boundary_conditions, boundary_conditions_movingmesh,
                                           uEltype)
end

function Base.show(io::IO, semi::SemidiscretizationMovingMesh)
    @nospecialize semi # reduce precompilation time

    print(io, "SemidiscretizationMovingMesh(")
    print(io, semi.mesh)
    print(io, ", ", semi.equations)
    print(io, ", ", semi.initial_condition)
    print(io, ", ", semi.boundary_conditions)
    print(io, ", ", semi.boundary_conditions_movingmesh)
    print(io, ", ", semi.source_terms)
    print(io, ", ", semi.solver)
    print(io, ", ", semi.solver_movingmesh)
    print(io, ", cache(")
    for (idx, key) in enumerate(keys(semi.cache))
        idx > 1 && print(io, " ")
        print(io, key)
    end
    print(io, "))")
end

function Base.show(io::IO, ::MIME"text/plain", semi::SemidiscretizationMovingMesh)
    @nospecialize semi # reduce precompilation time

    if get(io, :compact, false)
        show(io, semi)
    else
        summary_header(io, "SemidiscretizationMovingMesh")
        summary_line(io, "#spatial dimensions", ndims(semi.equations))
        summary_line(io, "mesh", semi.mesh)
        summary_line(io, "equations", semi.equations |> typeof |> nameof)
        summary_line(io, "initial condition", semi.initial_condition)

        summary_line(io, "boundary conditions", typeof(semi.boundary_conditions))
        # print_boundary_conditions(io, semi)

        summary_line(io, "source terms", semi.source_terms)
        summary_line(io, "solver", semi.solver |> typeof |> nameof)
        summary_line(io, "solver_movingmesh", semi.solver_movingmesh |> typeof |> nameof)
        summary_line(io, "total #DOFs per field", ndofs(semi))
        summary_footer(io)
    end
end

@inline Base.ndims(semi::SemidiscretizationMovingMesh) = ndims(semi.mesh)

@inline nvariables(semi::SemidiscretizationMovingMesh) = nvariables(semi.equations)

@inline Base.real(semi::SemidiscretizationMovingMesh) = real(semi.solver)

# @inline function ndofs(mesh, solver_movingmesh <: SolverMovingMesh, cache_movingmesh)
    # ndofs(mesh, solver_movingmesh, cache_movingmesh)
# end

@inline function ndofs(semi::SemidiscretizationMovingMesh)
    mesh, _, solver, solver_movingmesh, cache, _ = mesh_equations_solver_cache(semi)
    nd = ndofs(mesh, solver, cache)
    println("ndofs:", nd)
    kokoo
    # + ndofs(mesh, solver, cache)
end


@inline function mesh_equations_solver_cache(semi::SemidiscretizationMovingMesh)
    @unpack mesh, equations, solver, solver_movingmesh, cache, cache_movingmesh = semi
    return mesh, equations, solver, solver_movingmesh, cache, cache_movingmesh
end

function calc_error_norms(func, u_ode, t, analyzer, semi::SemidiscretizationMovingMesh,
                          cache_analysis)
    @unpack mesh, equations, initial_condition, solver, cache = semi
    u = wrap_array(u_ode, mesh, equations, solver, cache)

    calc_error_norms(func, u, t, analyzer, mesh, equations, initial_condition, solver,
                     cache, cache_analysis)
end

function compute_coefficients(t, semi::SemidiscretizationMovingMesh)
    compute_coefficients(semi.initial_condition, t, semi)
end

function allocate_coefficients(mesh::AbstractMesh, equations, dg::DG, solver_movingmesh, cache, cache_movingmesh)
    zeros(eltype(cache.elements),
          (1+nvariables(equations)) * nnodes(dg)^ndims(mesh) * nelements(dg, cache))
end

function compute_coefficients!(u_ode, func, t, mesh::AbstractMesh, equations, dg::DG, solver_movingmesh, cache, cache_movingmesh)
    # split into JU and J
    Ju = @view u_ode[0]
    compute_coefficients!(Ju, func, t, mesh, equations, dg, cache)
    # compute_coefficients!(u_ode, semi.initial_condition, t, semi)
end

@inline function wrap_array(u_ode::AbstractVector, mesh::AbstractMesh, equations,
                            dg::DGSEM, solver_movingmesh, cache, cache_moving_mesh)
    @boundscheck begin
        @assert length(u_ode) ==
        (1+nvariables(equations)) * nnodes(dg)^ndims(mesh) * nelements(dg, cache)
    end
    # We would like to use
    #     reshape(u_ode, (nvariables(equations), ntuple(_ -> nnodes(dg), ndims(mesh))..., nelements(dg, cache)))
    # but that results in
    #     ERROR: LoadError: cannot resize array with shared data
    # when we resize! `u_ode` during AMR.
    #
    # !!! danger "Segfaults"
    #     Remember to `GC.@preserve` temporaries such as copies of `u_ode`
    #     and other stuff that is only used indirectly via `wrap_array` afterwards!

    # Currently, there are problems when AD is used with `PtrArray`s in broadcasts
    # since LoopVectorization does not support `ForwardDiff.Dual`s. Hence, we use
    # optimized `PtrArray`s whenever possible and fall back to plain `Array`s
    # otherwise.
    if LoopVectorization.check_args(u_ode)
        # This version using `PtrArray`s from StrideArrays.jl is very fast and
        # does not result in allocations.
        #
        # !!! danger "Heisenbug"
        #     Do not use this code when `@threaded` uses `Threads.@threads`. There is
        #     a very strange Heisenbug that makes some parts very slow *sometimes*.
        #     In fact, everything can be fast and fine for many cases but some parts
        #     of the RHS evaluation can take *exactly* (!) five seconds randomly...
        #     Hence, this version should only be used when `@threaded` is based on
        #     `@batch` from Polyester.jl or something similar. Using Polyester.jl
        #     is probably the best option since everything will be handed over to
        #     Chris Elrod, one of the best performance software engineers for Julia.
        PtrArray(pointer(u_ode),
                 (StaticInt(1+nvariables(equations)),
                  ntuple(_ -> StaticInt(nnodes(dg)), ndims(mesh))...,
                  nelements(dg, cache)))
        #  (nvariables(equations), ntuple(_ -> nnodes(dg), ndims(mesh))..., nelements(dg, cache)))
    else
        # The following version is reasonably fast and allows us to `resize!(u_ode, ...)`.
        unsafe_wrap(Array{eltype(u_ode), ndims(mesh) + 2}, pointer(u_ode),
                    (1+nvariables(equations), ntuple(_ -> nnodes(dg), ndims(mesh))...,
                     nelements(dg, cache)))
    end
end


function rhs!(du_ode, u_ode, semi::SemidiscretizationMovingMesh, t)
    @unpack mesh, equations, initial_condition, boundary_conditions, boundary_conditions_movingmesh, source_terms, solver, solver_movingmesh, cache, cache_movingmesh = semi

    u = wrap_array(u_ode, mesh, equations, solver, solver_movingmesh, cache, cache_movingmesh)
    du = wrap_array(du_ode, mesh, equations, solver, solver_movingmesh, cache, cache_movingmesh)

    # TODO: Taal decide, do we need to pass the mesh?
    time_start = time_ns()
    @trixi_timeit timer() "rhs!" rhs!(du, u, t, mesh, equations, initial_condition,
                                      boundary_conditions, boundary_conditions_movingmesh,
                                      source_terms, solver, solver_movingmesh,
                                      cache, cache_movingmesh)
    runtime = time_ns() - time_start
    put!(semi.performance_counter, runtime)

    return nothing
end

include("dg_movingmesh.jl")

end # @muladd
