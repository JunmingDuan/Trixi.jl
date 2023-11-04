# By default, Julia/LLVM does not use fused multiply-add operations (FMAs).
# Since these FMAs can increase the performance of many numerical algorithms,
# we need to opt-in explicitly.
# See https://ranocha.de/blog/Optimizing_EC_Trixi for further details.
@muladd begin
#! format: noindent

function apply_smoothing!(mesh::StructuredMesh{2}, alpha, alpha_tmp, dg, cache)
    # Diffuse alpha values by setting each alpha to at least 50% of neighboring elements' alpha
    # Copy alpha values such that smoothing is indpedenent of the element access order
    nx, ny = size(mesh)
    alpha = reshape(alpha, nx, ny)
    alpha_tmp = reshape(alpha_tmp, nx, ny)
    alpha_tmp .= alpha

    # println("alpha")
    # for j = 1:ny
      # for i = 1:nx
        # print(alpha[i,j], " ")
      # end
      # println("\n")
    # end
    # kokoko

    @threaded for j = 2:ny-1, i = 2:nx-1 # inner part
      alpha[i,j] = 0.25*alpha_tmp[i,j] + 0.125*(alpha_tmp[i-1,j]+alpha_tmp[i+1,j]+alpha_tmp[i,j-1]+alpha_tmp[i,j+1]) + 0.0625*(alpha_tmp[i-1,j-1]+alpha_tmp[i+1,j-1]+alpha_tmp[i-1,j+1]+alpha_tmp[i+1,j+1])
    end

    # if assert isperiodic(mesh)
    # else
end

@inline function calc_indicator_gradient!(indicator_gradient, u,
      element, mesh::StructuredMesh{2},
                                                   equations, dg, cache)

  @unpack alpha_max, alpha_min, alpha_scale, alpha_smooth, variable = indicator_gradient
    @unpack alpha, alpha_tmp, indicator_threaded, modal_threaded, modal_tmp1_threaded = indicator_gradient.cache
    @unpack derivative_matrix = dg.basis

    indicator = indicator_threaded[Threads.threadid()]
    modal = modal_threaded[Threads.threadid()]
    modal_tmp1 = modal_tmp1_threaded[Threads.threadid()]

    # Calculate indicator variables at Gauss-Lobatto nodes
    for j in eachnode(dg), i in eachnode(dg)
        u_local = get_node_vars(u, equations, dg, i, j, element)
        indicator[i, j] = indicator_gradient.variable(u_local, equations)
    end

    x_der = derivative_matrix * indicator
    y_der = indicator * derivative_matrix'

    @. indicator = (x_der^2 + y_der^2)

    # Convert to modal representation
    multiply_scalar_dimensionwise!(modal, dg.basis.inverse_vandermonde_legendre, indicator, modal_tmp1)
    alpha[element] = modal[1,1]

    # alpha = reshape(alpha, 50, 50)
    # for j = 1:50, i = 1:50
      # if abs(i-35) < 10 && abs(j-35) < 10
        # alpha[i,j] = 20.0
      # else
        # alpha[i,j] = 1.0
      # end
    # end

    return modal[1,1]
end

function create_cache(::Type{IndicatorGradient},
                      equations::AbstractEquations{2}, basis::LobattoLegendreBasis)
    alpha = Vector{real(basis)}()
    alpha_tmp = similar(alpha)

    A = Array{real(basis), ndims(equations)}
    indicator_threaded = [A(undef, nnodes(basis), nnodes(basis))
                          for _ in 1:Threads.nthreads()]
    modal_threaded = [A(undef, nnodes(basis), nnodes(basis))
                      for _ in 1:Threads.nthreads()]
    modal_tmp1_threaded = [A(undef, nnodes(basis), nnodes(basis))
                           for _ in 1:Threads.nthreads()]

    return (; alpha, alpha_tmp, indicator_threaded, modal_threaded, modal_tmp1_threaded)
end

# this method is used when the indicator is constructed as for AMM
function create_cache(typ::Type{IndicatorGradient}, mesh,
                      equations::AbstractEquations{2}, dg::DGSEM, cache)
    create_cache(typ, equations, dg.basis)
end

end # @muladd

