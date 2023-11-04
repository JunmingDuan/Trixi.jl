# By default, Julia/LLVM does not use fused multiply-add operations (FMAs).
# Since these FMAs can increase the performance of many numerical algorithms,
# we need to opt-in explicitly.
# See https://ranocha.de/blog/Optimizing_EC_Trixi for further details.
@muladd begin
#! format: noindent

function (amm_callback::AMMCallback)(u_ode::AbstractVector, mesh::StructuredMesh{2},
                                     equations, dg::DG, cache, semi,
                                     t, iter;
                                     passive_args = ())
    @unpack controller, adaptor = amm_callback

    println("do something 3")
    u = wrap_array(u_ode, mesh, equations, dg, cache)
    @trixi_timeit timer() "indicator" controller(u, mesh, equations, dg, cache,
                                                 t = t, iter = iter)
    println("finish op here")
end

end # @muladd
