# By default, Julia/LLVM does not use fused multiply-add operations (FMAs).
# Since these FMAs can increase the performance of many numerical algorithms,
# we need to opt-in explicitly.
# See https://ranocha.de/blog/Optimizing_EC_Trixi for further details.
@muladd begin
#! format: noindent

struct AMMController{Indicator, Cache}
    max_iteration::Int
    indicator::Indicator
    cache::Cache
end

function AMMController(semi, indicator; max_iteration = 5)
    max_iteration = max_iteration
    cache = create_cache(AMMController, semi)
    AMMController{typeof(indicator), typeof(cache)}(max_iteration,
                                                    indicator,
                                                    cache)
end

function create_cache(indicator_type::Type{AMMController}, semi)
    create_cache(indicator_type, mesh_equations_solver_cache(semi)...)
end

function Base.show(io::IO, controller::AMMController)
    @nospecialize controller # reduce precompilation time

    print(io, "AMMController(")
    print(io, controller.indicator)
    print(io, ", max_iteration=", controller.max_iteration)
    print(io, ")")
end

function Base.show(io::IO, mime::MIME"text/plain", controller::AMMController)
    @nospecialize controller # reduce precompilation time

    if get(io, :compact, false)
        show(io, controller)
    else
        summary_header(io, "AMMController")
        summary_line(io, "indicator", controller.indicator |> typeof |> nameof)
        show(increment_indent(io), mime, controller.indicator)
        summary_line(io, "max_iteration", controller.max_iteration)
        summary_footer(io)
    end
end

function (controller::AMMController)(u::AbstractArray{<:Any},
                                            mesh, equations, dg::DG, cache;
                                            kwargs...)
    @unpack old_mesh_coordinates, new_mesh_coordinates = controller.cache

    alpha = controller.indicator(u, mesh, equations, dg, cache; kwargs...)

    println("mesh redistribution")

    nx, ny = size(mesh)
    alpha = reshape(alpha, nx, ny)

    node_coord = cache.elements.node_coordinates
    transfer_dgnodecoord_to_meshcoord!(mesh, node_coord, old_mesh_coordinates)
    redistribute!(mesh, alpha, controller.max_iteration, old_mesh_coordinates, new_mesh_coordinates)
    transfer_meshcoord_to_dgnodecoord!(mesh, dg, new_mesh_coordinates, node_coord)
end

include("amm_controllers_2d.jl")

end # @muladd
