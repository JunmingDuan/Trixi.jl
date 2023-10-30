
using OrdinaryDiffEq
using Trixi
using Plots

###############################################################################
# semidiscretization of the compressible Euler equations

equations = CompressibleEulerEquations2D(1.4)

function initial_condition_vortex(x, t, equations::CompressibleEulerEquations2D)
  r2 = x[1]^2 + x[2]^2
  s  = 1.0
  γ = equations.gamma
  strength = 5.0
  T  = 1.0 - (γ - 1.0)*strength^2/(8.0*γ*pi^2)*exp(1.0 - r2)
  k0 = strength/(2.0*pi)*exp(0.5*(1.0 - r2))
  rho    = (T/s)^(1.0/(γ - 1.0))
  rho_v1 = rho * (1.0 - k0*x[2])
  rho_v2 = rho * (1.0 + k0*x[1])
  rho_e  = T*rho/(γ - 1.0) + 0.5*(rho_v1^2 + rho_v2^2)/rho
  return SVector(rho, rho_v1, rho_v2, rho_e)
end

initial_condition = initial_condition_vortex

function mapping(xi_, eta_)
  R   = 5.0
  xi  = R * xi_
  eta = R * eta_
  x = xi  + 0.2 * sin(2.0 * pi * eta/R)
  y = eta + 0.2 * sin(2.0 * pi * xi/R)
  return SVector(x, y)
end

cells_per_dimension = (3, 4)

mesh = StructuredMesh(cells_per_dimension, mapping)

solver = DGSEM(polydeg=3, surface_flux=flux_lax_friedrichs)

semi = SemidiscretizationHyperbolic(mesh, equations, initial_condition, solver)


###############################################################################
# ODE solvers, callbacks etc.

tspan = (0.0, 10.0)
ode = semidiscretize(semi, tspan)

summary_callback = SummaryCallback()

analysis_interval = -1
analysis_callback = AnalysisCallback(semi, interval=analysis_interval)

alive_callback = AliveCallback(alive_interval=10)

save_solution = SaveSolutionCallback(interval=-1,
                                     save_initial_solution=true,
                                     save_final_solution=true,
                                     solution_variables=cons2prim)

stepsize_callback = StepsizeCallback(cfl=1.0)

visualization = VisualizationCallback(interval=20, show_mesh=true)

callbacks = CallbackSet(summary_callback,
                        analysis_callback, alive_callback,
                        save_solution,
                        stepsize_callback)

###############################################################################
# run the simulation

sol = solve(ode, CarpenterKennedy2N54(williamson_condition=false),
            dt=1.0, # solve needs some value here but it will be overwritten by the stepsize_callback
            save_everystep=false, callback=callbacks);
summary_callback() # print the timer summary

# pd = ScalarPlotData2D(sol[1], semi)
# pd = plot(sol[1], title="Error in density")
pd = PlotData2D(sol)
plot(pd["rho"], seriescolor = :heat)
plot!(getmesh(pd))

using Trixi2Vtk
trixi2vtk(joinpath("out", "solution_000000.h5"), output_directory="out")
