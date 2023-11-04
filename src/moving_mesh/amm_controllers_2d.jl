# By default, Julia/LLVM does not use fused multiply-add operations (FMAs).
# Since these FMAs can increase the performance of many numerical algorithms,
# we need to opt-in explicitly.
# See https://ranocha.de/blog/Optimizing_EC_Trixi for further details.
@muladd begin
#! format: noindent

function transfer_dgnodecoord_to_meshcoord!(mesh::StructuredMesh{2}, node_coord, mesh_coord)
  nx, ny = size(mesh)
  node_coord = reshape(node_coord, 2, size(node_coord,2), size(node_coord,3), nx, ny)

    @threaded for j = 1:ny+1, i = 1:nx+1
      if i < nx+1 && j < ny+1
        mesh_coord[:,i,j] = node_coord[:,1,1,i,j]
      elseif j < ny+1
        mesh_coord[:,i,j] = node_coord[:,end,1,i-1,j]
      elseif i < nx+1
        mesh_coord[:,i,j] = node_coord[:,1,end,i,j-1]
      else
        mesh_coord[:,i,j] = node_coord[:,end,end,i-1,j-1]
      end
      # println("i,j: ", i, ", ", j, "; ", mesh_coord[1,i,j], ", ", mesh_coord[2,i,j])
    end
end

function bilinear_interpolation(x, y, x1, x2, x3, x4)
    # x1: Bottom left, x2: Bottom right, x3: Top left, x4: Top right
    return 0.25 * (x1 * (1 - x) * (1 - y) +
            x2 * (1 + x) * (1 - y) +
            x3 * (1 - x) * (1 + y) +
            x4 * (1 + x) * (1 + y))
end

function transfer_meshcoord_to_dgnodecoord!(mesh::StructuredMesh{2}, dg::DG, mesh_coord, node_coord)
  nx, ny = size(mesh)
  nnodes = size(node_coord,2)
  # println("before")
  # println(node_coord)
  node_coord = reshape(node_coord, 2, size(node_coord,2), size(node_coord,3), nx, ny)

    @unpack nodes = dg.basis

    @threaded for j = 1:ny, i = 1:nx
      for l in eachnode(dg.basis), m in eachnode(dg.basis)
        node_coord[:,l,m,i,j] = bilinear_interpolation(nodes[l], nodes[m], mesh_coord[:,i,j], mesh_coord[:,i+1,j], mesh_coord[:,i,j+1], mesh_coord[:,i+1,j+1])
      end
      # if i < nx+1 && j < ny+1
        # mesh_coord[:,i,j] = node_coord[:,1,1,i,j]
      # elseif j < ny+1
        # mesh_coord[:,i,j] = node_coord[:,end,1,i-1,j]
      # elseif i < nx+1
        # mesh_coord[:,i,j] = node_coord[:,1,end,i,j-1]
      # else
        # mesh_coord[:,i,j] = node_coord[:,end,end,i-1,j-1]
      # end
      # println("i,j: ", i, ", ", j, "; ", mesh_coord[1,i,j], ", ", mesh_coord[2,i,j])
    end
    node_coord = reshape(node_coord, 2, size(node_coord,2), size(node_coord,3), nx*ny)
  # println("after")
    # println(node_coord)
    # kokoko
end

# this method is called when an `AMMController` is constructed
function create_cache(::Type{AMMController},
                      mesh::Union{StructuredMesh{2}}, equations,
                      dg::DG, cache)
    node_coord = cache.elements.node_coordinates
    ndim, nx, ny = size(node_coord,1), size(mesh,1), size(mesh,2)
    old_mesh_coordinates = Array{eltype(node_coord)}(undef, ndim, nx+1, ny+1)
    new_mesh_coordinates = similar(old_mesh_coordinates)

    return (; old_mesh_coordinates, new_mesh_coordinates)
end

function redistribute!(mesh::StructuredMesh{2}, alpha, max_iteration, old_mesh_coordinates, new_mesh_coordinates)
    nx, ny = size(mesh)
    dx, dy = 2/nx, 2/ny
    iter_mesh_coordinates = similar(old_mesh_coordinates)
    new_mesh_coordinates  .= old_mesh_coordinates

    # for j = 1:ny, i = 1:nx
      # if abs(i-35) < 10 && abs(j-35) < 10
        # alpha[i,j] = 20.0
      # else
        # alpha[i,j] = 1.0
      # end
    # end

    for _ in 1:max_iteration
      iter_mesh_coordinates .= new_mesh_coordinates
      # println("before")
      # println(iter_mesh_coordinates[1,:,:])
      @threaded for j = 2:ny, i = 2:nx # inner part
        alpha_w = 0.5*(alpha[i-1,j-1] + alpha[i-1,j])/dx^2
        alpha_e = 0.5*(alpha[i,j-1] + alpha[i,j])/dx^2
        alpha_s = 0.5*(alpha[i-1,j-1] + alpha[i,j-1])/dy^2
        alpha_n = 0.5*(alpha[i-1,j] + alpha[i,j])/dy^2
        new_mesh_coordinates[:,i,j] = (alpha_w*iter_mesh_coordinates[:,i-1,j] + alpha_e*iter_mesh_coordinates[:,i+1,j]
                                       + alpha_s*iter_mesh_coordinates[:,i,j-1] + alpha_n*iter_mesh_coordinates[:,i,j+1]) / (alpha_w + alpha_e + alpha_s + alpha_n)
        # println("i,j; w,e,s,n: ", i, " ", j, "; ", alpha_w, " ", alpha_e, " ", alpha_s, " ", alpha_n)
      end
      # kokok
      # println("after")
      # println(new_mesh_coordinates[1,:,:])
    end

end

end # @muladd
