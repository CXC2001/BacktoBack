using Unfold
using UnfoldMakie, CairoMakie
using UnfoldSim
dat, evts = UnfoldSim.predef_eeg(; noiselevel = 10, return_epoched = true)

f = @formula 0 ~ 1 + condition + continuous
designDict = Dict(Any => (f, range(0, 1, length = size(dat, 1))))

# #---
# se_solver = (x, y) -> Unfold.solver_default(x, y, stderror = true)
# m = Unfold.fit(UnfoldModel, designDict, evts, dat, solver = se_solver)
# results = coeftable(m)
# plot_erp(results; stderror = true)
# #---

# using Krylov, CUDA # necessary to load the right package extension
# gpu_solver =(x, y) -> Unfold.solver_krylov(x, y; GPU = true)
# m = Unfold.fit(UnfoldModel, designDict, evts, dat, solver = gpu_solver)

# using RobustModels # necessary to load the Unfold package extension
# se_solver = (x, y) -> Unfold.solver_robust(x, y)
# m = Unfold.fit(UnfoldModel, designDict, evts, dat, solver = se_solver)
# results = coeftable(m)
# plot_erp(results; stderror = true)

#---
b2b_solver = (x, y) -> Unfold.solver_b2b(x, y; cross_val_reps = 5)
dat_3d = permutedims(repeat(dat, 1, 1, 20), [3 1 2])
m = Unfold.fit(UnfoldModel, designDict, evts, dat_3d; solver = b2b_solver)
results = coeftable(m)
#---

plot_erp(results)