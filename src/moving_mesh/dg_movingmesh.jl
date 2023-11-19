# By default, Julia/LLVM does not use fused multiply-add operations (FMAs).
# Since these FMAs can increase the performance of many numerical algorithms,
# we need to opt-in explicitly.
# See https://ranocha.de/blog/Optimizing_EC_Trixi for further details.
@muladd begin
#! format: noindent

# This method is called when a SemidiscretizationHyperbolicMovingMesh is constructed.
# It constructs the basic `cache` used throughout the simulation to compute
# the RHS etc.
function create_cache_movingmesh(mesh::StructuredMesh{NDIMS}, dg::DG, RealT, uEltype) where {NDIMS}

    nelements = prod(size(mesh))
    temporal_contravariant_vectors = Array{RealT, NDIMS + 2}(undef, NDIMS,
                                                    ntuple(_ -> nnodes(dg.basis), NDIMS)...,
                                                    nelements)

    cache = (; temporal_contravariant_vectors)

    return cache
end

end # @muladd
